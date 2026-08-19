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
    ratio_min: torch.Tensor
    ratio_max: torch.Tensor
    ratio_abs_deviation_p95: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "total_loss": float(self.total_loss.detach().item()),
            "policy_loss": float(self.policy_loss.detach().item()),
            "raw_kl_loss": float(self.raw_kl_loss.detach().item()),
            "weighted_kl_loss": float(self.weighted_kl_loss.detach().item()),
            "approx_kl": float(self.approx_kl.detach().item()),
            "clip_fraction": float(self.clip_fraction.detach().item()),
            "ratio_mean": float(self.ratio_mean.detach().item()),
            "ratio_min": float(self.ratio_min.detach().item()),
            "ratio_max": float(self.ratio_max.detach().item()),
            "ratio_abs_deviation_p95": float(
                self.ratio_abs_deviation_p95.detach().item()
            ),
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

    batch = log_probs.numel()
    if batch == 0 or old_log_probs.numel() != batch:
        raise ValueError("new and old log-probabilities must share a non-empty batch")
    if advantages.numel() not in {1, batch} or noise_weights.numel() not in {1, batch}:
        raise ValueError("advantages and noise weights must be scalar or match log-probabilities")
    if policy_means.shape != reference_means.shape:
        raise ValueError("policy/reference transition means must have the same shape")
    if clip_range < 0 or kl_beta < 0:
        raise ValueError("clip_range and kl_beta must be non-negative")
    vectors = [log_probs, old_log_probs, advantages, noise_weights]
    if not all(torch.isfinite(value).all() for value in vectors):
        raise FloatingPointError("non-finite PPO inputs")

    advantages = advantages.reshape(-1)
    noise_weights = noise_weights.reshape(-1)
    if advantages.numel() == 1:
        advantages = advantages.expand(batch)
    if noise_weights.numel() == 1:
        noise_weights = noise_weights.expand(batch)

    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - float(clip_range), 1.0 + float(clip_range))
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    policy_loss = -(noise_weights * surrogate).mean()

    stds = transition_stds.reshape(-1)
    if torch.any(stds <= 0) or not torch.isfinite(stds).all():
        raise FloatingPointError("transition std must be finite and positive")
    if stds.numel() == 1:
        per_sample_kl = (policy_means - reference_means).float().square().mean().reshape(1)
    elif policy_means.ndim > 0 and policy_means.shape[0] == stds.numel():
        reduce_dims = tuple(range(1, policy_means.ndim))
        per_sample_kl = (
            (policy_means - reference_means).float().square().mean(dim=reduce_dims)
        )
    else:
        raise ValueError("transition std must be scalar or match the KL rollout batch")
    per_sample_kl = per_sample_kl / (2.0 * stds.square())
    raw_kl_loss = per_sample_kl.mean()
    weighted_kl_loss = float(kl_beta) * raw_kl_loss
    total_loss = policy_loss + weighted_kl_loss
    approx_kl = 0.5 * log_ratio.square().mean()
    ratio_abs_deviation = torch.abs(ratio - 1.0)
    clip_fraction = (ratio_abs_deviation > float(clip_range)).float().mean()
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
        ratio_min=ratio.min(),
        ratio_max=ratio.max(),
        ratio_abs_deviation_p95=torch.quantile(ratio_abs_deviation.detach().float(), 0.95),
    )
