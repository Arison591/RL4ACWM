from __future__ import annotations

from collections.abc import Sequence


def leave_one_out_advantages(rewards: Sequence[float]) -> tuple[list[float], list[float]]:
    """Return per-item LOO baselines and advantages for one valid seed group."""
    if len(rewards) < 2:
        raise ValueError("leave-one-out advantage requires at least two rollouts")
    total = float(sum(rewards))
    denominator = len(rewards) - 1
    baselines = [(total - float(reward)) / denominator for reward in rewards]
    return baselines, [float(reward) - baseline for reward, baseline in zip(rewards, baselines)]


def local_leave_one_out_advantages(
    local_rewards: Sequence[float], global_rewards: Sequence[float]
) -> list[float]:
    """Compute local advantages against the complete distributed seed group."""
    if len(global_rewards) < 2:
        raise ValueError("global leave-one-out advantage requires at least two rewards")
    total = float(sum(global_rewards))
    denominator = len(global_rewards) - 1
    return [float(reward) - (total - float(reward)) / denominator for reward in local_rewards]
