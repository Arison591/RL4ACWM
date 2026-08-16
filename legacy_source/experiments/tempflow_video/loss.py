from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TempFlowLossOutput:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    raw_kl_loss: torch.Tensor
    weighted_kl_loss: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_mean: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "total_loss": float(self.total_loss.detach().item()),
            "policy_loss": float(self.policy_loss.detach().item()),
            "raw_kl_loss": float(self.raw_kl_loss.detach().item()),
            "weighted_kl_loss": float(self.weighted_kl_loss.detach().item()),
            "approx_kl": float(self.approx_kl.detach().item()),
            "clip_fraction": float(self.clip_fraction.detach().item()),
            "ratio_mean": float(self.ratio_mean.detach().item()),
        }


def tempflow_grpo_loss(
    *,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    noise_weights: torch.Tensor,
    policy_means: torch.Tensor,
    reference_means: torch.Tensor,
    transition_stds: torch.Tensor,
    clip_range: float,
    kl_beta: float,
) -> TempFlowLossOutput:
    """Clipped TempFlow-GRPO objective plus closed-form reference KL.

    There is deliberately no flow-matching MSE term: rollout actions are scored
    through their actual Gaussian transition log probabilities.
    """

    vectors = [log_probs, old_log_probs, advantages, noise_weights]
    batch = log_probs.numel()
    if batch == 0 or any(value.numel() != batch for value in vectors):
        raise ValueError("log-probability, advantage and weight vectors must share a non-empty batch")
    if policy_means.shape != reference_means.shape or policy_means.shape[0] != batch:
        raise ValueError("policy/reference transition means must share the rollout batch")
    if transition_stds.numel() not in {1, batch}:
        raise ValueError("transition std must be scalar or one value per rollout")
    if clip_range < 0 or kl_beta < 0:
        raise ValueError("clip_range and kl_beta must be non-negative")
    if not all(torch.isfinite(value).all() for value in vectors):
        raise FloatingPointError("non-finite PPO inputs")

    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - float(clip_range), 1.0 + float(clip_range))
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    policy_loss = -(noise_weights * surrogate).mean()

    stds = transition_stds.reshape(-1)
    if stds.numel() == 1:
        stds = stds.expand(batch)
    if torch.any(stds <= 0) or not torch.isfinite(stds).all():
        raise FloatingPointError("transition std must be finite and positive")
    reduce_dims = tuple(range(1, policy_means.ndim))
    per_sample_kl = (policy_means - reference_means).square().mean(dim=reduce_dims)
    per_sample_kl = per_sample_kl / (2.0 * stds.square())
    raw_kl_loss = per_sample_kl.mean()
    weighted_kl_loss = float(kl_beta) * raw_kl_loss
    total_loss = policy_loss + weighted_kl_loss
    approx_kl = 0.5 * log_ratio.square().mean()
    clip_fraction = (torch.abs(ratio - 1.0) > float(clip_range)).float().mean()
    if not torch.isfinite(total_loss):
        raise FloatingPointError("TempFlow total loss is non-finite")
    return TempFlowLossOutput(
        total_loss=total_loss,
        policy_loss=policy_loss,
        raw_kl_loss=raw_kl_loss,
        weighted_kl_loss=weighted_kl_loss,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        ratio_mean=ratio.mean(),
    )
