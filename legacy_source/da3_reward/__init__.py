"""Standalone DA3 GT-guided pseudo-depth reward package."""

from .reward import (
    DA3ModelRunner,
    DA3RewardConfig,
    DA3RewardResult,
    compute_da3_gt_reward,
    load_da3_cache,
    make_gt_motion_mask,
    median_mad_advantage,
    save_da3_cache,
)
from .mask import DA3DynamicMaskConfig, make_da3_dynamic_mask

__all__ = [
    "DA3DynamicMaskConfig",
    "DA3ModelRunner",
    "DA3RewardConfig",
    "DA3RewardResult",
    "compute_da3_gt_reward",
    "load_da3_cache",
    "make_gt_motion_mask",
    "make_da3_dynamic_mask",
    "median_mad_advantage",
    "save_da3_cache",
]
