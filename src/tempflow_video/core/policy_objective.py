from __future__ import annotations

from dataclasses import dataclass

import torch

from .reference_kl import closed_form_equal_variance_kl


@dataclass(frozen=True)
class PolicyLoss:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    raw_kl_loss: torch.Tensor
    weighted_kl_loss: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_mean: torch.Tensor


def grpo_surrogate(log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                   advantages: torch.Tensor, noise_weights: torch.Tensor,
                   clip_range: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    loss = -(noise_weights * torch.minimum(ratio * advantages, clipped * advantages)).mean()
    return loss, 0.5 * log_ratio.square().mean(), (torch.abs(ratio - 1) > clip_range).float().mean(), ratio.mean()


def legacy_policy_objective(*, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                            advantages: torch.Tensor, noise_weights: torch.Tensor,
                            policy_means: torch.Tensor, reference_means: torch.Tensor,
                            transition_stds: torch.Tensor, clip_range: float,
                            kl_beta: float) -> PolicyLoss:
    policy, approx, fraction, ratio = grpo_surrogate(log_probs, old_log_probs, advantages,
                                                      noise_weights, clip_range)
    raw_kl = closed_form_equal_variance_kl(policy_means, reference_means, transition_stds)
    weighted_kl = kl_beta * raw_kl
    return PolicyLoss(policy + weighted_kl, policy, raw_kl, weighted_kl, approx, fraction, ratio)

