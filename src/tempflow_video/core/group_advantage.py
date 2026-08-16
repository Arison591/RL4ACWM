from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class GroupAdvantage:
    rewards: torch.Tensor
    advantages: torch.Tensor
    mean: float
    std: float
    skipped: bool


def standardize_group(rewards: Sequence[float] | torch.Tensor, *, epsilon: float = 1e-6,
                      min_std: float = 1e-8, clip: float | None = None) -> GroupAdvantage:
    values = torch.as_tensor(rewards, dtype=torch.float64).flatten()
    if values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("a finite group of at least two rewards is required")
    mean, std = values.mean(), values.std(unbiased=False)
    skipped = bool(std.item() < float(min_std))
    advantage = torch.zeros_like(values) if skipped else (values - mean) / (std + epsilon)
    if clip is not None:
        advantage = advantage.clamp(-float(clip), float(clip))
    return GroupAdvantage(values, advantage, float(mean), float(std), skipped)

