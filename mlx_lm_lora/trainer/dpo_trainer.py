import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.callbacks import TrainingCallback
from tqdm import tqdm

from .sft_trainer import SFTTrainingArgs, grad_checkpoint


@dataclass
class DPOTrainingArgs(SFTTrainingArgs):
    beta: float = field(
        default=0.1, metadata={"help": "Temperature parameter for DPO training."}
    )
    loss_type: str = field(
        default="sigmoid",
        metadata={
            "help": "DPO loss type: 'sigmoid', 'hinge', 'ipo', 'dpop', or 'length_normalized'."
        },
    )
    delta: float = field(
        default=50.0, metadata={"help": "Delta parameter for DPOP loss type."}
    )
    reference_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to reference model weights. If None, uses the same model."
        },
    )


def get_token_scores(model, x, mask):
    # Per-token score index i = log p(targets[i] | inputs[i]) = log p(x[i+1] | x[i]).
    # The score at index i predicts the TARGET token at absolute position i+1, so it is
    # valid iff that target token is real (not padding). We therefore mask with the
    # TARGET-position slice ``mask[:, 1:]`` (mask[i+1]), NOT the source slice
    # ``mask[:, :-1]`` (mask[i]).
    #
    # This matches AllenAI open-instruct's reference DPO exactly:
    #   dpo_utils._get_batch_logps: loss_mask = labels[:, 1:] != -100  (TARGET position)
    #   applied to per_token_logps[:, :-1] where per_token_logps[i] = log p(labels[i+1]|x_i).
    # (open-instruct/open_instruct/dpo_utils.py:706-712, calculate_per_token_logps at
    #  padding_free_collator.py:10-16). The reference is authoritative.
    #
    # The prior ``mask[:, :-1]`` (SOURCE position) wrongly counted the last-real→first-pad
    # prediction on every PADDED row (~11 nats/row), diverging from both the reference and
    # the torch backend (which already masks at ``attention_mask[:, 1:]``). On equal-length
    # un-padded rows the two slices coincide, so this is a no-op there. Fixed in ReAlign
    # #1505; the DPO stage had not yet run under the divergent mask (SFT-only bake), so no
    # ladder checkpoint is affected.
    inputs, targets = x[:, :-1], x[:, 1:]
    logits = model(inputs).astype(mx.float32)
    return -nn.losses.cross_entropy(logits, targets) * mask[:, 1:]


def compute_score(scores, mask, loss_type):
    token_count = mask.sum(-1)
    if loss_type in ("ipo", "length_normalized"):
        return scores.sum(-1) / token_count
    return scores.sum(-1)


def dpo_loss(
    policy_chosen_score: mx.array,
    policy_rejected_score: mx.array,
    reference_chosen_score: mx.array,
    reference_rejected_score: mx.array,
    chosen_masks: mx.array,
    rejected_masks: mx.array,
    beta: float,
    delta: float,
    loss_type: str = "sigmoid",
):
    # Preference logits
    logits = (policy_chosen_score - policy_rejected_score) - (
        reference_chosen_score - reference_rejected_score
    )

    # Loss calculation
    if loss_type == "sigmoid":
        losses = -nn.log_sigmoid(beta * logits)
    elif loss_type == "hinge":
        losses = nn.relu(1 - beta * logits)
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "dpop":
        penalty = mx.maximum(
            mx.zeros_like(policy_chosen_score),
            reference_chosen_score - policy_chosen_score,
        )
        losses = -(nn.log_sigmoid(beta * logits) - delta * penalty)
    elif loss_type == "length_normalized":
        # Length-normalized DPO: compute_score (lines 45-49) already divided
        # the per-token sums by token count for this loss_type, so the upstream
        # `logits` variable is the difference of *per-token-averaged* log-probs
        # rather than raw sequence sums. β operates on a smaller-magnitude
        # signal here, so the useful β is approximately an order of magnitude
        # larger than vanilla sigmoid DPO; e.g. AllenAI Tulu-3 uses β=5.0
        # with length-normalized (loss_type: dpo_norm, beta: 5 in
        # external/reference/open-instruct/configs/train_configs/tulu3/tulu3_dpo_8b.yaml)
        # vs the conventional β≈0.1 with sigmoid.
        losses = -nn.log_sigmoid(beta * logits)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Token counts and rewards
    num_chosen_tokens = chosen_masks.sum(-1)
    num_rejected_tokens = rejected_masks.sum(-1)
    num_tokens = (num_chosen_tokens + num_rejected_tokens).sum()

    chosen_reward = beta * mx.mean(policy_chosen_score - reference_chosen_score)
    rejected_reward = beta * mx.mean(policy_rejected_score - reference_rejected_score)
    reward = mx.stack([chosen_reward, rejected_reward])

    # Metrics
    metrics = {
        "accuracies": mx.mean((chosen_reward > rejected_reward).astype(mx.float32)),
        "margins": mx.mean(chosen_reward - rejected_reward),
        "policy_rejected_logps": mx.mean(policy_rejected_score / num_rejected_tokens),
        "policy_chosen_logps": mx.mean(policy_chosen_score / num_chosen_tokens),
        "rejected_logits_mean": mx.mean(policy_rejected_score),
        "chosen_logits_mean": mx.mean(policy_chosen_score),
    }

    mx.clear_cache()
    return mx.mean(losses), reward, num_tokens, metrics


def _completion_prompt_len(chosen_ids, rejected_ids, item=None):
    """Resolve the PROMPT boundary (# leading tokens to exclude from the loss).

    AllenAI open-instruct's reference DPO supervises ONLY completion tokens: it sets
    ``labels[:len(prompt)] = -100`` and reduces with ``loss_mask = labels[:, 1:] != -100``
    (dpo_utils.py:706-712) — the prompt tokens are dropped from BOTH the numerator and the
    dpo_norm divisor. ReAlign #1506 brings the MLX path to that convention.

    Prompt length is resolved in priority order:
      1. ``item["prompt_len"]`` when the dataset carries an explicit boundary (the
         ReAlign-owned ``DPODataset`` records the true rendered prompt-only token count).
      2. The longest common prefix of the chosen and rejected token id lists. For a
         preference pair both sequences are ``prompt + assistant-header + completion`` with
         an IDENTICAL prefix, so the first divergent token IS the completion boundary. This
         needs no separate prompt tokenization and — because it operates on the SAME
         rendered+tokenized sequences the model sees — is immune to the BPE prompt/completion
         seam-merge class of bug (ReAlign OLMo2 boundary gotchas). Its only imprecision is
         when both completions happen to share a leading token, which symmetrically trims a
         token or two off both completions — negligible for the length-normalized average.
    """
    if item is not None:
        p = item.get("prompt_len")
        if p is not None:
            return int(p)
    n = min(len(chosen_ids), len(rejected_ids))
    i = 0
    while i < n and chosen_ids[i] == rejected_ids[i]:
        i += 1
    return i


def iterate_dpo_batches(
    dataset,
    batch_size,
    max_seq_length,
    train=False,
    return_indices: bool = False,
    mask_prompt: bool = False,
):
    """Iterate over DPO preference batches.

    Args:
        dataset: List-like of dicts with ``chosen`` / ``rejected`` token id lists.
        batch_size: Effective per-step micro batch size (pre-distributed).
        max_seq_length: Cap for token sequences in the padded tensors.
        train: When ``True``, shuffle batch order each epoch and loop forever.
        return_indices: When ``True`` (default ``False``), additionally yield a
            5th tensor with the GLOBAL dataset indices for the current batch.
            This is required by the reference-logprob precompute path which
            needs to address scores by JSONL-row, not by per-batch slot
            (mlx-lm-lora `iterate_dpo_batches` sorts by `len(chosen)` so the
            slot-position is not stable across `batch_size` choices). Default
            is ``False`` to preserve backward compatibility with all existing
            callers (upstream tests, ``evaluate_dpo``).
        mask_prompt: When ``True``, the emitted chosen/rejected masks are
            COMPLETION-ONLY loss masks (prompt tokens zeroed) rather than the
            full attention mask — the AllenAI ``dpo_norm`` reference convention
            (ReAlign #1506). The prompt boundary per row is resolved by
            ``_completion_prompt_len`` (explicit ``prompt_len`` or chosen/rejected
            common prefix). Default ``False`` yields the historical full-token
            attention mask, byte-for-byte identical to the pre-#1506 behaviour so
            every existing caller (SFT parity, upstream tests) is unaffected.
    """
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]["chosen"]))

    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("Batch size must be divisible by workers")

    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    while True:
        indices = (
            np.random.permutation(len(batch_idx)) if train else range(len(batch_idx))
        )
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]

            # Get and process lengths
            chosen_lengths = [len(x["chosen"]) for x in batch]
            rejected_lengths = [len(x["rejected"]) for x in batch]
            max_length = min(
                max(max(chosen_lengths), max(rejected_lengths)), max_seq_length
            )

            # Dynamic padding based on batch content
            max_length_in_batch = max_length

            chosen_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)
            rejected_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)

            chosen_masks = np.zeros(
                (batch_size // step, max_length_in_batch), np.float32
            )
            rejected_masks = np.zeros(
                (batch_size // step, max_length_in_batch), np.float32
            )

            for j in range(batch_size // step):
                chosen_length = min(chosen_lengths[j], max_seq_length)
                rejected_length = min(rejected_lengths[j], max_seq_length)

                chosen_arr[j, :chosen_length] = batch[j]["chosen"][:chosen_length]
                rejected_arr[j, :rejected_length] = batch[j]["rejected"][
                    :rejected_length
                ]

                chosen_masks[j, :chosen_length] = 1.0
                rejected_masks[j, :rejected_length] = 1.0

                if mask_prompt:
                    # Completion-only loss mask (AllenAI dpo_norm, #1506): zero the
                    # prompt prefix so it leaves BOTH the numerator and the divisor.
                    # Clamp so >= 1 completion token survives per row (a non-zero
                    # length_normalized divisor; mirrors the reference clamp).
                    prompt_len = _completion_prompt_len(
                        batch[j]["chosen"], batch[j]["rejected"], item=batch[j]
                    )
                    c_cut = min(prompt_len, max(chosen_length - 1, 0))
                    r_cut = min(prompt_len, max(rejected_length - 1, 0))
                    chosen_masks[j, :c_cut] = 0.0
                    rejected_masks[j, :r_cut] = 0.0

            if return_indices:
                yield (
                    mx.array(chosen_arr),
                    mx.array(rejected_arr),
                    mx.array(chosen_masks),
                    mx.array(rejected_masks),
                    mx.array(batch_idx[i], dtype=mx.int32),
                )
            else:
                yield mx.array(chosen_arr), mx.array(rejected_arr), mx.array(
                    chosen_masks
                ), mx.array(rejected_masks)

        if not train:
            break


def evaluate_dpo(
    model,
    ref_model,
    dataset,
    batch_size,
    num_batches,
    beta: float,
    delta: float,
    max_seq_length,
    loss_type,
    loss_fn: callable = dpo_loss,
    ref_score_fn=None,
    mask_prompt: bool = False,
):
    """Evaluate DPO loss / rewards over ``dataset``.

    Args:
        ref_score_fn: Optional callable ``(batch_indices: mx.array) -> (
            chosen_ref_score: mx.array, rejected_ref_score: mx.array)``. When
            provided, the per-batch reference forward pass is skipped and
            cached scores are gathered by global JSONL index. ``ref_model`` is
            ignored when ``ref_score_fn`` is not ``None``. Default ``None``
            preserves the existing live-ref behaviour.
        mask_prompt: When ``True``, score COMPLETION tokens only (AllenAI
            ``dpo_norm`` reference convention, #1506) — threaded to
            ``iterate_dpo_batches``. Default ``False`` = historical full-token
            behaviour.
    """
    all_losses = 0
    all_rewards = mx.zeros((2,))
    all_metrics = None
    ntokens = 0

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    use_cached_ref = ref_score_fn is not None
    for _, batch in zip(
        index_iterator,
        iterate_dpo_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            return_indices=use_cached_ref,
            mask_prompt=mask_prompt,
        ),
    ):
        if use_cached_ref:
            chosen, rejected, chosen_masks, rejected_masks, batch_indices = batch
        else:
            chosen, rejected, chosen_masks, rejected_masks = batch
            batch_indices = None

        policy_chosen_scores = get_token_scores(model, chosen, chosen_masks)
        policy_rejected_scores = get_token_scores(model, rejected, rejected_masks)

        policy_chosen_score = compute_score(
            policy_chosen_scores, chosen_masks, loss_type
        )
        policy_rejected_score = compute_score(
            policy_rejected_scores, rejected_masks, loss_type
        )

        if use_cached_ref:
            reference_chosen_score, reference_rejected_score = ref_score_fn(
                batch_indices
            )
        elif ref_model is None:
            reference_chosen_score = mx.zeros_like(policy_chosen_score)
            reference_rejected_score = mx.zeros_like(policy_rejected_score)
        else:
            ref_chosen_scores = mx.stop_gradient(
                get_token_scores(ref_model, chosen, chosen_masks)
            )
            ref_rejected_scores = mx.stop_gradient(
                get_token_scores(ref_model, rejected, rejected_masks)
            )
            reference_chosen_score = compute_score(
                ref_chosen_scores, chosen_masks, loss_type
            )
            reference_rejected_score = compute_score(
                ref_rejected_scores, rejected_masks, loss_type
            )

        loss_value, reward, toks, metrics = loss_fn(
            policy_chosen_score=policy_chosen_score,
            policy_rejected_score=policy_rejected_score,
            reference_chosen_score=reference_chosen_score,
            reference_rejected_score=reference_rejected_score,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            loss_type=loss_type,
            beta=beta,
            delta=delta,
        )
        all_losses += loss_value * toks
        all_rewards += reward
        ntokens += toks

        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks

        mx.eval(all_losses, all_rewards, ntokens)
    all_losses = mx.distributed.all_sum(all_losses)
    all_rewards = mx.distributed.all_sum(all_rewards)
    ntokens = mx.distributed.all_sum(ntokens)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_rewards = (all_rewards / ntokens).tolist()
    avg_loss = (all_losses / ntokens).item()

    return avg_loss, avg_rewards, ntokens, avg_metrics


def train_dpo(
    model,
    ref_model,
    optimizer,
    train_dataset,
    val_dataset,
    args: DPOTrainingArgs = DPOTrainingArgs(),
    loss_fn: callable = dpo_loss,
    training_callback: TrainingCallback = None,
    loss_type="sigmoid",
    use_compile: bool = True,
    ref_score_fn=None,
    mask_prompt: bool = False,
):
    """Run DPO training over ``train_dataset``.

    Args:
        model: Trainable policy model (mx.nn.Module).
        ref_model: Frozen reference model. May be ``None`` for a zero baseline
            (debug only — produces a degenerate gradient).
        optimizer: An MLX optimizer instance bound to ``model``.
        train_dataset: Iterable preference dataset.
        val_dataset: Validation dataset (same shape as ``train_dataset``).
        args: ``DPOTrainingArgs`` controlling iters, batch size, beta, etc.
        loss_fn: Loss function with the ``dpo_loss`` signature.
        training_callback: Optional ``TrainingCallback``.
        loss_type: One of ``"sigmoid"``, ``"hinge"``, ``"ipo"``, ``"dpop"``,
            ``"length_normalized"``.
        use_compile: If ``True`` (default), wrap the per-step graph in
            ``mx.compile``. **Set ``False`` for full-parameter fine-tuning.**

    Notes:
        ``mx.compile`` traces and caches the per-step graph together with the
        captured ``state`` (model + optimizer). For LoRA the trainable subgraph
        is small and the in-place ``optimizer.update(model, grad)`` re-binds
        correctly on every call. For full-parameter fine-tuning the in-place
        update is silently skipped on subsequent calls — the LR counter
        advances (it lives in ``optimizer.state`` inside the traced state) but
        model weights never change. Symptoms: saved adapters are byte-identical
        to the source checkpoint, ``final_loss`` converges to
        ``ln(2) ≈ 0.6931`` (policy ≡ ref → reward = β·0 → -log σ(0)) and all
        reward channels are ``0.0``.

        Pass ``use_compile=False`` for full-parameter fine-tuning. Reproducer
        and regression test:
        ``tests/integration/test_dpo_actually_learns.py`` in the realign repo
        (Issue #990).
    """
    mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    tqdm.write(f"Starting training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        tqdm.write(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    grad_accum_steps = args.gradient_accumulation_steps
    if grad_accum_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    state = [model.state, optimizer.state, mx.random.state]

    use_cached_ref = ref_score_fn is not None

    def loss_wrapper(
        chosen,
        rejected,
        chosen_masks,
        rejected_masks,
        reference_chosen_score=None,
        reference_rejected_score=None,
    ):
        policy_chosen_scores = get_token_scores(model, chosen, chosen_masks)
        policy_rejected_scores = get_token_scores(model, rejected, rejected_masks)

        policy_chosen_score = compute_score(
            policy_chosen_scores, chosen_masks, loss_type
        )
        policy_rejected_score = compute_score(
            policy_rejected_scores, rejected_masks, loss_type
        )

        if reference_chosen_score is not None and reference_rejected_score is not None:
            # Cached-ref path — pre-gathered scores are passed in by caller.
            pass
        elif ref_model is None:
            reference_chosen_score = mx.zeros_like(policy_chosen_score)
            reference_rejected_score = mx.zeros_like(policy_rejected_score)
        else:
            ref_chosen_scores = mx.stop_gradient(
                get_token_scores(ref_model, chosen, chosen_masks)
            )
            ref_rejected_scores = mx.stop_gradient(
                get_token_scores(ref_model, rejected, rejected_masks)
            )
            reference_chosen_score = compute_score(
                ref_chosen_scores, chosen_masks, loss_type
            )
            reference_rejected_score = compute_score(
                ref_rejected_scores, rejected_masks, loss_type
            )

        return loss_fn(
            policy_chosen_score=policy_chosen_score,
            policy_rejected_score=policy_rejected_score,
            reference_chosen_score=reference_chosen_score,
            reference_rejected_score=reference_rejected_score,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            beta=args.beta,
            delta=args.delta,
            loss_type=loss_type,
        )

    loss_value_and_grad = nn.value_and_grad(model, loss_wrapper)

    def _step_impl(batch, prev_grad, do_update):
        if use_cached_ref:
            (
                chosen,
                rejected,
                chosen_masks,
                rejected_masks,
                ref_chosen_score,
                ref_rejected_score,
            ) = batch
            (lvalue, reward, toks, metrics), grad = loss_value_and_grad(
                chosen,
                rejected,
                chosen_masks,
                rejected_masks,
                ref_chosen_score,
                ref_rejected_score,
            )
        else:
            chosen, rejected, chosen_masks, rejected_masks = batch
            (lvalue, reward, toks, metrics), grad = loss_value_and_grad(
                chosen, rejected, chosen_masks, rejected_masks
            )

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if args.gradient_accumulation_steps > 1:
                grad = tree_map(lambda x: x / args.gradient_accumulation_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, reward, toks, metrics, grad

    # ``state`` (defined above) is captured by reference; calling
    # ``partial(mx.compile, inputs=state, outputs=state)(_step_impl)`` is
    # equivalent to the decorator form ``@partial(mx.compile, ...)``.
    if use_compile:
        step = partial(mx.compile, inputs=state, outputs=state)(_step_impl)
    else:
        step = _step_impl

    losses = 0
    rewards = mx.zeros((2,))
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    accumulated_metrics = {
        "accuracies": 0,
        "margins": 0,
        "policy_rejected_logps": 0,
        "policy_chosen_logps": 0,
        "rejected_logits_mean": 0,
        "chosen_logits_mean": 0,
    }
    grad_accum = None

    start = time.perf_counter()
    pbar = tqdm(range(1, args.iters + 1), desc="Training", disable=rank != 0)
    for it in pbar:
        raw_batch = next(
            iterate_dpo_batches(
                dataset=train_dataset,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                train=True,
                return_indices=use_cached_ref,
                mask_prompt=mask_prompt,
            )
        )
        if use_cached_ref:
            (
                _chosen,
                _rejected,
                _chosen_masks,
                _rejected_masks,
                _batch_indices,
            ) = raw_batch
            ref_chosen_score, ref_rejected_score = ref_score_fn(_batch_indices)
            batch = (
                _chosen,
                _rejected,
                _chosen_masks,
                _rejected_masks,
                ref_chosen_score,
                ref_rejected_score,
            )
        else:
            batch = raw_batch

        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss, val_rewards, val_ntokens, val_metrics = evaluate_dpo(
                model=model,
                ref_model=ref_model,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                loss_fn=loss_fn,
                beta=args.beta,
                delta=args.delta,
                loss_type=loss_type,
                ref_score_fn=ref_score_fn,
                mask_prompt=mask_prompt,
            )
            val_time = time.perf_counter() - stop
            if rank == 0:
                tqdm.write(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val chosen reward {val_rewards[0]:.3f}, "
                    f"Val rejected reward {val_rewards[1]:.3f}, "
                    f"Val accuracy {val_metrics['accuracies']:.3f}, "
                    f"Val margin {val_metrics['margins']:.3f}, "
                    f"Val took {val_time:.3f}s",
                )

            if training_callback is not None:
                training_callback.on_val_loss_report(
                    {
                        "iteration": it,
                        "val_loss": val_loss,
                        "val_chosen_reward": val_rewards[0],
                        "val_rejected_reward": val_rewards[1],
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "val_time": val_time,
                    }
                )

            start = time.perf_counter()

        lvalue, reward, toks, metrics, grad_accum = step(
            batch,
            grad_accum,
            it % grad_accum_steps == 0,
        )
        losses += lvalue
        rewards += reward
        n_tokens += toks
        steps += 1

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        mx.eval(state, losses, rewards, n_tokens, grad_accum)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
            train_rewards = mx.distributed.all_sum(rewards).tolist()
            train_rewards = [r / (steps * world_size) for r in train_rewards]
            avg_metrics = {
                k: v / (steps * world_size) for k, v in accumulated_metrics.items()
            }
            n_tokens = mx.distributed.all_sum(n_tokens).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9

            if rank == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{train_loss:.3f}",
                        "it/s": f"{it_sec:.3f}",
                    }
                )
                tqdm.write(
                    f"\nIter {it}: "
                    f"loss {train_loss:.3f}, "
                    f"chosen_r {train_rewards[0]:.3f}, "
                    f"rejected_r {train_rewards[1]:.3f}, "
                    f"acc {avg_metrics['accuracies']:.3f}, "
                    f"margin {avg_metrics['margins']:.3f}, "
                    f"lr {learning_rate:.3e}, "
                    f"it/s {it_sec:.3f}, "
                    f"tok/s {tokens_sec:.3f}, "
                    f"peak_mem {peak_mem:.3f}GB"
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "train_chosen_reward": train_rewards[0],
                    "train_rejected_reward": train_rewards[1],
                    **{f"train_{k}": v for k, v in avg_metrics.items()},
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            rewards = mx.zeros((2,))
            n_tokens = 0
            steps = 0
            accumulated_metrics = {k: 0 for k in accumulated_metrics}
            start = time.perf_counter()

        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            tqdm.write(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    tqdm.write(f"Saved final weights to {args.adapter_file}.")
