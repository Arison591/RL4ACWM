from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class TrainingNoiseLevel:
    level_index: int
    scheduler_index: int
    scheduler_timestep: float
    sigma: float
    flow_time: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def build_training_noise_levels(scheduler: Any, num_levels: int) -> tuple[TrainingNoiseLevel, ...]:
    if num_levels <= 0:
        raise ValueError("num_levels must be positive")
    sigmas = torch.as_tensor(scheduler.sigmas, dtype=torch.float64).flatten()
    timesteps = torch.as_tensor(scheduler.timesteps, dtype=torch.float64).flatten()
    usable = min(len(timesteps), len(sigmas))
    positive = [index for index in range(usable) if float(sigmas[index]) > 0.0]
    if len(positive) < num_levels:
        raise ValueError(f"scheduler only has {len(positive)} positive sigma levels")
    positions = torch.linspace(0, len(positive) - 1, num_levels).round().long().tolist()
    indices = [positive[position] for position in positions]
    if len(set(indices)) != num_levels:
        raise ValueError("noise-level selection produced duplicates")
    result = []
    for level, index in enumerate(indices):
        sigma = float(sigmas[index])
        result.append(TrainingNoiseLevel(level, index, float(timesteps[index]), sigma, sigma / (sigma + 1.0)))
    return tuple(result)


def validate_base_probabilities(probabilities: Sequence[float], num_levels: int) -> tuple[float, ...]:
    if len(probabilities) != num_levels or any(value <= 0 for value in probabilities):
        raise ValueError("base probabilities must match levels and have full support")
    total = float(sum(probabilities))
    if not torch.isfinite(torch.tensor(total)) or total <= 0:
        raise ValueError("base probabilities must be finite")
    return tuple(float(value) / total for value in probabilities)
