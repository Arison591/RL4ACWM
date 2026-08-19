from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from tempflow_video.core.group_advantage import GroupAdvantage, standardize_group


@dataclass(frozen=True)
class ComponentAdvantages:
    action: GroupAdvantage
    psnr: GroupAdvantage
    both_components_skipped: bool

    def metrics(self) -> dict[str, float]:
        return {"action_reward_mean": self.action.mean, "action_reward_std": self.action.std,
                "action_advantage_mean": float(self.action.advantages.mean()),
                "action_advantage_std": float(self.action.advantages.std(unbiased=False)),
                "psnr_advantage_mean": float(self.psnr.advantages.mean()),
                "psnr_advantage_std": float(self.psnr.advantages.std(unbiased=False)),
                "action_component_skipped": float(self.action.skipped),
                "psnr_component_skipped": float(self.psnr.skipped),
                "both_components_skipped": float(self.both_components_skipped)}


def component_advantages(action_rewards: Sequence[float], psnr_future_db: Sequence[float], *,
                         action_min_group_std: float | None, psnr_min_group_std_db: float | None,
                         advantage_clip: float = 1.0, epsilon: float = 1e-6,
                         formal_training: bool = True) -> ComponentAdvantages:
    if formal_training and (action_min_group_std is None or psnr_min_group_std_db is None):
        raise RuntimeError("formal training requires calibrated component std thresholds")
    action_threshold = 0.0 if action_min_group_std is None else action_min_group_std
    psnr_threshold = 0.0 if psnr_min_group_std_db is None else psnr_min_group_std_db
    action = standardize_group(action_rewards, epsilon=epsilon, min_std=action_threshold, clip=advantage_clip)
    psnr = standardize_group(psnr_future_db, epsilon=epsilon, min_std=psnr_threshold, clip=advantage_clip)
    return ComponentAdvantages(action, psnr, action.skipped and psnr.skipped)

