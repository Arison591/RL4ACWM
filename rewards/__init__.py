"""Reward modules used by RL training."""

from .any4d_adapter import (
    Any4DAdapterConfig,
    Any4DGeometryAdapter,
    load_official_any4d_model,
)
from .geometry_adapter import GeometryAdapter, GeometryOutput, MockGeometryAdapter
from .rgb_temporal_consistency import compute_rgb_temporal_error
from .vggrpo_reward import (
    CameraMotionRewardConfig,
    CameraMotionRewardOutput,
    GeometryRewardOutput,
    GeometryRewardConfig,
    GroupNormalizationConfig,
    GroupNormalizationOutput,
    RewardOutput,
    SingleViewGeometryReward,
    SingleViewCameraMotionReward,
    VGGRPORewardComponents,
    assemble_reward_output,
    compute_camera_motion_rewards,
    compute_geometry_rewards,
    compute_single_view_camera_motion_reward,
    compute_single_view_geometry_reward,
    compute_vggrpo_reward,
    compute_vggrpo_reward_components,
    normalize_and_combine_group_rewards,
)

__all__ = [
    "Any4DAdapterConfig",
    "Any4DGeometryAdapter",
    "load_official_any4d_model",
    "GeometryAdapter",
    "GeometryOutput",
    "CameraMotionRewardConfig",
    "CameraMotionRewardOutput",
    "GeometryRewardConfig",
    "GeometryRewardOutput",
    "GroupNormalizationConfig",
    "GroupNormalizationOutput",
    "RewardOutput",
    "MockGeometryAdapter",
    "SingleViewGeometryReward",
    "SingleViewCameraMotionReward",
    "VGGRPORewardComponents",
    "assemble_reward_output",
    "compute_camera_motion_rewards",
    "compute_geometry_rewards",
    "compute_rgb_temporal_error",
    "compute_single_view_camera_motion_reward",
    "compute_single_view_geometry_reward",
    "compute_vggrpo_reward",
    "compute_vggrpo_reward_components",
    "normalize_and_combine_group_rewards",
]
