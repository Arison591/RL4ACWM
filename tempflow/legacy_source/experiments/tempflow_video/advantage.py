from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


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
