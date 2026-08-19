import pytest
import torch

from experiments.tempflow_video.loss import tempflow_grpo_loss


def test_tempflow_loss_uses_transition_ratio_and_closed_form_kl():
    log_probs = torch.tensor([-1.0, -1.2], requires_grad=True)
    old_log_probs = log_probs.detach().clone()
    advantages = torch.tensor([1.0, -1.0])
    weights = torch.tensor([1.5, 0.5])
    means = torch.zeros(2, 2, 3, requires_grad=True)
    references = torch.zeros_like(means)
    output = tempflow_grpo_loss(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        noise_weights=weights,
        policy_means=means,
        reference_means=references,
        transition_stds=torch.tensor([0.4, 0.4]),
        clip_range=0.2,
        kl_beta=0.01,
    )

    assert output.policy_loss.item() == pytest.approx(-0.5)
    assert output.raw_kl_loss.item() == 0.0
    assert output.approx_kl.item() == 0.0
    assert output.clip_fraction.item() == 0.0
    output.total_loss.backward()
    assert torch.isfinite(log_probs.grad).all()


def test_reference_kl_is_positive_after_policy_mean_moves():
    means = torch.ones(2, 2, 2, requires_grad=True)
    output = tempflow_grpo_loss(
        log_probs=torch.zeros(2, requires_grad=True),
        old_log_probs=torch.zeros(2),
        advantages=torch.tensor([1.0, -1.0]),
        noise_weights=torch.ones(2),
        policy_means=means,
        reference_means=torch.zeros_like(means),
        transition_stds=torch.tensor([0.5]),
        clip_range=0.2,
        kl_beta=0.1,
    )

    assert output.raw_kl_loss.item() == pytest.approx(2.0)
    assert output.weighted_kl_loss.item() == pytest.approx(0.2)
