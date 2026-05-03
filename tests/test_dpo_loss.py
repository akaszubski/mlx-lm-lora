"""Unit tests for dpo_loss, including the new length_normalized branch."""

import mlx.core as mx

from mlx_lm_lora.trainer.dpo_trainer import dpo_loss


def test_length_normalized_loss_returns_finite_loss_and_expected_shapes():
    # Per-token-averaged log-prob scores (compute_score already normalized by
    # token count when loss_type == "length_normalized"). 2 examples.
    policy_chosen = mx.array([-0.5, -0.3])
    policy_rejected = mx.array([-0.9, -1.1])
    reference_chosen = mx.array([-0.6, -0.4])
    reference_rejected = mx.array([-0.8, -1.0])

    # 2 examples × 4 tokens, all unmasked
    chosen_masks = mx.ones((2, 4))
    rejected_masks = mx.ones((2, 4))

    loss, reward, num_tokens, metrics = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        chosen_masks,
        rejected_masks,
        beta=5.0,
        delta=50.0,
        loss_type="length_normalized",
    )

    assert mx.isfinite(loss).item() is True
    assert reward.shape == (2,)
    expected_keys = {
        "accuracies",
        "margins",
        "policy_rejected_logps",
        "policy_chosen_logps",
        "rejected_logits_mean",
        "chosen_logits_mean",
    }
    assert set(metrics.keys()) == expected_keys
