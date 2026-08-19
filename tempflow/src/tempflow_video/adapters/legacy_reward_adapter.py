from __future__ import annotations

from typing import Any

import torch

from tempflow_video.runtime.upstream_loader import import_upstream


class LegacyRewardAdapter:
    """Exact forwarding boundary for the unchanged upstream Action/legacy joint reward."""

    def __init__(self, reward_config: dict[str, Any], compute_fn=None):
        self.config = dict(reward_config)
        self.compute_fn = compute_fn or import_upstream("experiments.awm_coca.reward_runner").compute_head_reward

    @torch.no_grad()
    def score(self, gt_path: str, pred_path: str, **kwargs) -> dict[str, Any]:
        return self.compute_fn(gt_path, pred_path, **kwargs)

