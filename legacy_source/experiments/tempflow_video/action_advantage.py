"""Noise-gated, component-wise Action GRPO advantages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from experiments.tempflow_video.advantage import standardize_group_rewards


COMPONENTS = ("command", "fdce", "iou")


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2:
        return float("nan")
    # Branch rewards are continuous in the intended path; deterministic ordinal
    # ranks are sufficient for a diagnostic and avoid an optional scipy runtime.
    left_rank = torch.argsort(torch.argsort(left)).double()
    right_rank = torch.argsort(torch.argsort(right)).double()
    left_std = left_rank.std(unbiased=False)
    right_std = right_rank.std(unbiased=False)
    if left_std == 0 or right_std == 0:
        return float("nan")
    return float(((left_rank - left_rank.mean()) * (right_rank - right_rank.mean())).mean() / (left_std * right_std))


@dataclass(frozen=True)
class ActionAdvantageResult:
    advantages: torch.Tensor | None
    skip_reason: str | None
    metrics: dict[str, float]


def build_action_advantages(
    *,
    command_raw_error: Sequence[float],
    fdce_reward: Sequence[float],
    iou_reward: Sequence[float],
    valid_arms: Sequence[int],
    coverage: Sequence[float],
    noise_floors: Mapping[str, float],
    weights: Mapping[str, float],
    epsilon: float = 1.0e-6,
) -> ActionAdvantageResult:
    """Build one advantage per branch, or skip if command is not trustworthy.

    Command uses negative raw pixel error. FDCE and IoU inputs are already
    transformed reward components, hence all three input vectors are
    higher-is-better before standardization.
    """
    values = {
        "command": -torch.as_tensor(command_raw_error, dtype=torch.float64).flatten(),
        "fdce": torch.as_tensor(fdce_reward, dtype=torch.float64).flatten(),
        "iou": torch.as_tensor(iou_reward, dtype=torch.float64).flatten(),
    }
    size = values["command"].numel()
    if size < 2 or any(value.numel() != size for value in values.values()):
        raise ValueError("Action components must have equal group size >= 2")
    if any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError("Action components must be finite")
    complete = (
        len(valid_arms) == size
        and len(coverage) == size
        and all(int(item) == 2 for item in valid_arms)
        and all(float(item) >= 1.0 for item in coverage)
    )
    results = {
        name: standardize_group_rewards(
            value, epsilon=epsilon, zero_std_threshold=0.0
        )
        for name, value in values.items()
    }
    gates = {
        name: bool(
            complete if name == "command" else True
        )
        and results[name].reward_std > max(float(noise_floors[name]), epsilon)
        for name in COMPONENTS
    }
    metrics = {
        **{f"{name}_std": results[name].reward_std for name in COMPONENTS},
        **{f"{name}_noise_floor": float(noise_floors[name]) for name in COMPONENTS},
        **{f"{name}_gate": float(gates[name]) for name in COMPONENTS},
        "command_complete": float(complete),
    }
    if not gates["command"]:
        metrics["skip_reason_command_not_informative"] = 1.0
        return ActionAdvantageResult(None, "command_not_informative", metrics)

    contributions = {
        name: (
            float(weights[name]) * results[name].advantages
            if gates[name]
            else torch.zeros_like(results[name].advantages)
        )
        for name in COMPONENTS
    }
    total = sum(contributions.values())
    if not torch.isfinite(total).all() or abs(float(total.mean())) > 1.0e-10:
        raise FloatingPointError("component-wise Action advantage is invalid")
    if not (bool((total > 0).any()) and bool((total < 0).any())):
        raise FloatingPointError("component-wise Action advantage has no ranking")
    metrics.update({
        f"mean_abs_weighted_{name}": float(contributions[name].abs().mean())
        for name in COMPONENTS
    })
    metrics.update({
        f"spearman_{name}_total_advantage": _spearman(values[name], total)
        for name in COMPONENTS
    })
    metrics["spearman_command_fdce"] = _spearman(values["command"], values["fdce"])
    metrics["spearman_command_iou"] = _spearman(values["command"], values["iou"])
    return ActionAdvantageResult(total, None, metrics)
