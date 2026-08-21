from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch


def fuse_psnr_sobel_rewards(
    psnr_scores: Sequence[float] | torch.Tensor,
    sobel_scores: Sequence[float] | torch.Tensor,
    *,
    psnr_weight: float = 0.8,
    sobel_weight: float = 0.2,
    psnr_scale: float | None = None,
    sobel_scale: float | None = None,
    psnr_scale_floor: float = 0.005,
    sobel_scale_floor: float = 0.0005,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Scale two group components and fuse them before final GRPO standardization.

    A missing fixed scale uses the current group's population std with a floor.
    Group centering is intentionally omitted: the following GRPO
    standardization removes the same constant without changing rankings.
    """
    psnr = torch.as_tensor(psnr_scores, dtype=torch.float64).flatten()
    sobel = torch.as_tensor(sobel_scores, dtype=torch.float64).flatten()
    if psnr.numel() < 1 or psnr.shape != sobel.shape:
        raise ValueError("PSNR/Sobel fusion requires matching non-empty groups")
    if not torch.isfinite(psnr).all() or not torch.isfinite(sobel).all():
        raise ValueError("PSNR/Sobel group values must be finite")
    weights = (float(psnr_weight), float(sobel_weight))
    if any(not math.isfinite(value) or value < 0.0 for value in weights) or sum(weights) <= 0.0:
        raise ValueError("PSNR/Sobel weights must be finite, non-negative and not both zero")
    weight_sum = sum(weights)
    normalized_weights = (weights[0] / weight_sum, weights[1] / weight_sum)

    observed_psnr_std = float(psnr.std(unbiased=False).item())
    observed_sobel_std = float(sobel.std(unbiased=False).item())

    def resolve_scale(fixed: float | None, observed: float, floor: float, name: str) -> float:
        if not math.isfinite(float(floor)) or float(floor) <= 0.0:
            raise ValueError(f"{name}_scale_floor must be finite and positive")
        if fixed is None:
            return max(observed, float(floor))
        if not math.isfinite(float(fixed)) or float(fixed) <= 0.0:
            raise ValueError(f"{name}_scale must be null or finite and positive")
        return float(fixed)

    resolved_psnr_scale = resolve_scale(
        psnr_scale, observed_psnr_std, psnr_scale_floor, "psnr"
    )
    resolved_sobel_scale = resolve_scale(
        sobel_scale, observed_sobel_std, sobel_scale_floor, "sobel"
    )
    fused = (
        normalized_weights[0] * psnr / resolved_psnr_scale
        + normalized_weights[1] * sobel / resolved_sobel_scale
    )
    return fused, {
        "psnr_weight": normalized_weights[0],
        "sobel_weight": normalized_weights[1],
        "psnr_group_mean_db": float(psnr.mean().item()),
        "psnr_group_std_db": observed_psnr_std,
        "sobel_group_mean": float(sobel.mean().item()),
        "sobel_group_std": observed_sobel_std,
        "psnr_scale": resolved_psnr_scale,
        "sobel_scale": resolved_sobel_scale,
        "psnr_scale_source": "fixed" if psnr_scale is not None else "group_std_with_floor",
        "sobel_scale_source": "fixed" if sobel_scale is not None else "group_std_with_floor",
    }


@dataclass(frozen=True)
class GroupAdvantageResult:
    rewards: torch.Tensor
    advantages: torch.Tensor
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    advantage_mean: float
    advantage_std: float
    advantage_min: float
    advantage_max: float
    zero_std: bool

    def metrics(self) -> dict[str, float]:
        return {
            "group_reward_mean": self.reward_mean,
            "group_reward_std": self.reward_std,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "advantage_mean": self.advantage_mean,
            "advantage_std": self.advantage_std,
            "advantage_min": self.advantage_min,
            "advantage_max": self.advantage_max,
            "zero_std_group_ratio": float(self.zero_std),
        }


def standardize_group_rewards(
    rewards: Sequence[float] | torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
    zero_std_threshold: float = 1.0e-8,
) -> GroupAdvantageResult:
    """Population-standardize rewards from exactly one TempFlow branch group."""

    values = torch.as_tensor(rewards, dtype=torch.float64).flatten()
    if values.numel() < 2:
        raise ValueError("TempFlow group advantage requires at least two branches")
    if not torch.isfinite(values).all():
        raise ValueError("group rewards must be finite")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("advantage epsilon must be finite and positive")
    mean = values.mean()
    std = values.std(unbiased=False)
    zero_std = bool(std.item() <= float(zero_std_threshold))
    advantages = torch.zeros_like(values) if zero_std else (values - mean) / (std + epsilon)
    advantage_std = advantages.std(unbiased=False)
    return GroupAdvantageResult(
        rewards=values,
        advantages=advantages,
        reward_mean=float(mean.item()),
        reward_std=float(std.item()),
        reward_min=float(values.min().item()),
        reward_max=float(values.max().item()),
        advantage_mean=float(advantages.mean().item()),
        advantage_std=float(advantage_std.item()),
        advantage_min=float(advantages.min().item()),
        advantage_max=float(advantages.max().item()),
        zero_std=zero_std,
    )
