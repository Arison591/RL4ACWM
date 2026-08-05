"""VGGRPO-style camera-motion and geometry rewards for RGB model outputs."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .geometry_adapter import GeometryAdapter, GeometryOutput
from .rgb_temporal_consistency import compute_rgb_temporal_error


@dataclass(frozen=True)
class GeometryRewardConfig:
    """Configuration for geometry reprojection consistency."""

    num_views: int = 3
    topk_worst_frames: int = 3
    eps: float = 1.0e-8
    min_valid_projected_pixels: int = 128
    static_flow_threshold: float = 3.0e-2
    rgb_temporal_weight: float = 0.75
    static_error_scale: float = 2.0e-1
    rgb_resize_long_side: int = 160
    min_rgb_temporal_frames: int = 7

    def __post_init__(self) -> None:
        if self.num_views <= 0:
            raise ValueError("num_views must be positive")
        if self.topk_worst_frames <= 0:
            raise ValueError("topk_worst_frames must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.min_valid_projected_pixels <= 0:
            raise ValueError("min_valid_projected_pixels must be positive")
        if self.static_flow_threshold < 0:
            raise ValueError("static_flow_threshold must be non-negative")
        if not 0.0 <= self.rgb_temporal_weight <= 1.0:
            raise ValueError("rgb_temporal_weight must be in [0, 1]")
        if self.static_error_scale <= 0:
            raise ValueError("static_error_scale must be positive")
        if self.rgb_resize_long_side <= 0:
            raise ValueError("rgb_resize_long_side must be positive")
        if self.min_rgb_temporal_frames < 2:
            raise ValueError("min_rgb_temporal_frames must be at least 2")


@dataclass(frozen=True)
class CameraMotionRewardConfig:
    """Configuration for camera translation/rotation smoothness."""

    num_views: int = 3
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.num_views <= 0:
            raise ValueError("num_views must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


@dataclass
class SingleViewCameraMotionReward:
    """Camera smoothness reward for one candidate/view."""

    reward: Tensor
    valid: bool
    translation_error: Tensor
    rotation_error: Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraMotionRewardOutput:
    """Per-candidate and per-view camera-motion smoothness rewards."""

    motion_reward: Tensor
    per_view_reward: Tensor
    valid_mask: Tensor
    per_view_valid_mask: Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VGGRPORewardComponents:
    """Motion and geometry components computed from shared geometry inference."""

    motion: CameraMotionRewardOutput
    geometry: "GeometryRewardOutput"


@dataclass
class SingleViewGeometryReward:
    """Geometry reward and validity details for one candidate/view."""

    reward: Tensor
    valid: bool
    frame_errors: Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryRewardOutput:
    """Geometry rewards for a GRPO candidate group.

    For input ``[K, V, T, C, H, W]``, ``geometry_reward`` and
    ``valid_mask`` have shape ``[K]``, while per-view tensors have shape
    ``[K, V]``.  For input ``[B, K, V, T, C, H, W]``, the corresponding
    shapes are ``[B, K]`` and ``[B, K, V]``.
    """

    geometry_reward: Tensor
    per_view_reward: Tensor
    valid_mask: Tensor
    per_view_valid_mask: Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupNormalizationConfig:
    """Configuration for combining group-relative reward components."""

    eps: float = 1.0e-8
    min_valid_group_size: int = 2
    motion_weight: float = 0.1
    geometry_weight: float = 0.9

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.min_valid_group_size < 2:
            raise ValueError("min_valid_group_size must be at least 2")
        if self.motion_weight < 0 or self.geometry_weight < 0:
            raise ValueError("reward weights must be non-negative")
        if abs(self.motion_weight + self.geometry_weight - 1.0) > self.eps:
            raise ValueError("motion_weight and geometry_weight must sum to 1")


@dataclass
class GroupNormalizationOutput:
    """Separately normalized rewards and their combined advantage."""

    normalized_motion_reward: Tensor
    normalized_geometry_reward: Tensor
    advantage: Tensor
    valid_mask: Tensor
    valid_group_mask: Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardOutput:
    """Training-facing VGGRPO reward result.

    Raw and normalized component tensors use group layout ``[K]`` or
    ``[B, K]``.  Per-view component tensors append a final ``V`` dimension.
    ``advantage`` is ``None`` when no group is valid, which gives the training
    loop an explicit signal to skip the reward update.
    """

    motion_reward: Tensor
    geometry_reward: Tensor
    advantage: Optional[Tensor]
    valid_mask: Tensor
    valid_group_mask: Tensor
    normalized_motion_reward: Tensor
    normalized_geometry_reward: Tensor
    per_view_motion_reward: Optional[Tensor]
    per_view_motion_valid_mask: Optional[Tensor]
    per_view_geometry_reward: Tensor
    per_view_valid_mask: Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _invalid_camera_motion_reward(
    output: GeometryOutput,
    code: str,
    reason: str,
) -> SingleViewCameraMotionReward:
    details = {"invalid_code": code, "invalid_reason": reason}
    empty = output.camera_poses.new_empty((0,))
    return SingleViewCameraMotionReward(
        reward=output.camera_poses.new_tensor(float("nan")),
        valid=False,
        translation_error=empty,
        rotation_error=empty,
        diagnostics=details,
    )


def _so3_log_map(rotation_matrices: Tensor, eps: float) -> Tensor:
    """Return axis-angle vectors for rotation matrices shaped ``[N, 3, 3]``."""

    traces = rotation_matrices.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((traces - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    sin_axis = 0.5 * torch.stack(
        [
            rotation_matrices[:, 2, 1] - rotation_matrices[:, 1, 2],
            rotation_matrices[:, 0, 2] - rotation_matrices[:, 2, 0],
            rotation_matrices[:, 1, 0] - rotation_matrices[:, 0, 1],
        ],
        dim=-1,
    )
    omega = torch.empty_like(sin_axis)
    small = theta < 1.0e-4
    near_pi = (torch.pi - theta) < 1.0e-4
    regular = ~(small | near_pi)
    omega[small] = sin_axis[small]
    if regular.any():
        scale = theta[regular] / torch.sin(theta[regular]).clamp_min(eps)
        omega[regular] = sin_axis[regular] * scale.unsqueeze(-1)

    # The skew part vanishes at pi.  Recover its axis from the symmetric
    # matrix terms so exact/near-180-degree inputs remain finite and meaningful.
    for matrix_index in torch.nonzero(near_pi, as_tuple=False).flatten().tolist():
        matrix = rotation_matrices[matrix_index]
        diagonal = torch.diagonal(matrix)
        dominant_axis = int(torch.argmax(diagonal).item())
        axis = matrix.new_zeros(3)
        if dominant_axis == 0:
            axis[0] = torch.sqrt(((matrix[0, 0] + 1.0) / 2.0).clamp_min(eps))
            axis[1] = (matrix[0, 1] + matrix[1, 0]) / (4.0 * axis[0])
            axis[2] = (matrix[0, 2] + matrix[2, 0]) / (4.0 * axis[0])
        elif dominant_axis == 1:
            axis[1] = torch.sqrt(((matrix[1, 1] + 1.0) / 2.0).clamp_min(eps))
            axis[0] = (matrix[0, 1] + matrix[1, 0]) / (4.0 * axis[1])
            axis[2] = (matrix[1, 2] + matrix[2, 1]) / (4.0 * axis[1])
        else:
            axis[2] = torch.sqrt(((matrix[2, 2] + 1.0) / 2.0).clamp_min(eps))
            axis[0] = (matrix[0, 2] + matrix[2, 0]) / (4.0 * axis[2])
            axis[1] = (matrix[1, 2] + matrix[2, 1]) / (4.0 * axis[2])
        axis = axis / torch.linalg.vector_norm(axis).clamp_min(eps)
        omega[matrix_index] = theta[matrix_index] * axis
    return omega


@torch.no_grad()
def compute_single_view_camera_motion_reward(
    output: GeometryOutput,
    config: CameraMotionRewardConfig,
) -> SingleViewCameraMotionReward:
    """Compute the task-specified translation/rotation smoothness reward."""

    poses = output.camera_poses
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        return _invalid_camera_motion_reward(
            output,
            "invalid_camera_pose_shape",
            f"camera_poses must have shape [T, 4, 4], got {tuple(poses.shape)}",
        )
    if poses.shape[0] < 3:
        return _invalid_camera_motion_reward(
            output,
            "insufficient_motion_frames",
            f"only {poses.shape[0]} frames; need at least 3",
        )
    if not torch.isfinite(poses).all():
        return _invalid_camera_motion_reward(
            output,
            "nonfinite_camera_pose",
            "camera_poses contain NaN or Inf",
        )

    centers = poses[:, :3, 3]
    translation_velocity = centers[1:] - centers[:-1]
    translation_acceleration = (
        translation_velocity[1:] - translation_velocity[:-1]
    )
    translation_denominator = (
        torch.linalg.vector_norm(translation_velocity[1:], dim=-1)
        + torch.linalg.vector_norm(translation_velocity[:-1], dim=-1)
        + config.eps
    )
    translation_terms = torch.linalg.vector_norm(
        translation_acceleration, dim=-1
    ) / translation_denominator
    translation_error = translation_terms.mean()

    rotations = poses[:, :3, :3]
    relative_rotations = rotations[:-1].transpose(-1, -2) @ rotations[1:]
    angular_velocity = _so3_log_map(relative_rotations, config.eps)
    angular_acceleration = angular_velocity[1:] - angular_velocity[:-1]
    rotation_denominator = (
        torch.linalg.vector_norm(angular_velocity[1:], dim=-1)
        + torch.linalg.vector_norm(angular_velocity[:-1], dim=-1)
        + config.eps
    )
    rotation_terms = torch.linalg.vector_norm(
        angular_acceleration, dim=-1
    ) / rotation_denominator
    rotation_error = rotation_terms.mean()

    if not torch.isfinite(translation_error) or not torch.isfinite(rotation_error):
        return _invalid_camera_motion_reward(
            output,
            "nonfinite_motion_error",
            "camera motion smoothness produced NaN or Inf",
        )
    reward = 0.5 * (
        1.0 / (1.0 + translation_error)
        + 1.0 / (1.0 + rotation_error)
    )
    return SingleViewCameraMotionReward(
        reward=reward,
        valid=True,
        translation_error=translation_error,
        rotation_error=rotation_error,
        diagnostics={
            "translation_error": float(translation_error),
            "rotation_error": float(rotation_error),
            "translation_terms": translation_terms.detach().cpu().tolist(),
            "rotation_terms": rotation_terms.detach().cpu().tolist(),
        },
    )


@torch.no_grad()
def compute_camera_motion_rewards(
    videos: Tensor,
    adapter: GeometryAdapter,
    config: CameraMotionRewardConfig,
) -> CameraMotionRewardOutput:
    """Compute independent per-view motion rewards and their equal average."""

    if videos.ndim == 6:
        batched_videos = videos.unsqueeze(0)
        squeeze_batch = True
    elif videos.ndim == 7:
        batched_videos = videos
        squeeze_batch = False
    else:
        raise ValueError(
            "videos must have shape [K, V, T, C, H, W] or "
            f"[B, K, V, T, C, H, W], got {tuple(videos.shape)}"
        )

    batch_size, group_size, num_views = batched_videos.shape[:3]
    output_shape = (batch_size, group_size)
    per_view_shape = (batch_size, group_size, num_views)
    per_view_reward = torch.full(
        per_view_shape,
        float("nan"),
        dtype=torch.float32,
        device=videos.device,
    )
    per_view_valid_mask = torch.zeros(
        per_view_shape, dtype=torch.bool, device=videos.device
    )
    motion_reward = torch.full(
        output_shape,
        float("nan"),
        dtype=torch.float32,
        device=videos.device,
    )
    valid_mask = torch.zeros(output_shape, dtype=torch.bool, device=videos.device)
    diagnostics: Dict[str, Any] = {
        "input_shape": tuple(videos.shape),
        "invalid_views": [],
        "per_view": {},
    }

    if num_views != config.num_views:
        diagnostics["invalid_code"] = "view_count_mismatch"
        diagnostics["invalid_reason"] = (
            f"expected {config.num_views} views, got {num_views}"
        )
    else:
        for batch_index in range(batch_size):
            for candidate_index in range(group_size):
                for view_index in range(num_views):
                    key = (
                        f"batch{batch_index}/candidate{candidate_index}/"
                        f"view{view_index}"
                    )
                    video = batched_videos[
                        batch_index, candidate_index, view_index
                    ]
                    try:
                        geometry = adapter.infer(video)
                        result = compute_single_view_camera_motion_reward(
                            geometry, config
                        )
                    except Exception as error:
                        diagnostics["invalid_views"].append(
                            {
                                "batch_index": batch_index,
                                "candidate_index": candidate_index,
                                "view_index": view_index,
                                "code": "geometry_inference_failure",
                                "reason": (
                                    "geometry inference failed: "
                                    f"{type(error).__name__}: {error}"
                                ),
                            }
                        )
                        continue

                    diagnostics["per_view"][key] = result.diagnostics
                    if not result.valid:
                        diagnostics["invalid_views"].append(
                            {
                                "batch_index": batch_index,
                                "candidate_index": candidate_index,
                                "view_index": view_index,
                                "code": result.diagnostics.get(
                                    "invalid_code", "invalid_camera_motion"
                                ),
                                "reason": result.diagnostics.get(
                                    "invalid_reason", "unknown camera motion error"
                                ),
                            }
                        )
                        continue

                    per_view_reward[
                        batch_index, candidate_index, view_index
                    ] = result.reward.to(
                        device=videos.device, dtype=torch.float32
                    )
                    per_view_valid_mask[
                        batch_index, candidate_index, view_index
                    ] = True

                if bool(
                    per_view_valid_mask[batch_index, candidate_index].all().item()
                ):
                    motion_reward[batch_index, candidate_index] = per_view_reward[
                        batch_index, candidate_index
                    ].mean()
                    valid_mask[batch_index, candidate_index] = True

    diagnostics["valid_rate"] = float(valid_mask.float().mean().item())
    diagnostics["per_view_valid_rate"] = [
        float(per_view_valid_mask[..., view_index].float().mean().item())
        for view_index in range(num_views)
    ]
    invalid_reason_counts: Dict[str, int] = {}
    for invalid_view in diagnostics["invalid_views"]:
        code = invalid_view["code"]
        invalid_reason_counts[code] = invalid_reason_counts.get(code, 0) + 1
    if "invalid_code" in diagnostics:
        code = diagnostics["invalid_code"]
        invalid_reason_counts[code] = batch_size * group_size
    diagnostics["invalid_reason_counts"] = invalid_reason_counts

    if squeeze_batch:
        motion_reward = motion_reward.squeeze(0)
        per_view_reward = per_view_reward.squeeze(0)
        valid_mask = valid_mask.squeeze(0)
        per_view_valid_mask = per_view_valid_mask.squeeze(0)
    return CameraMotionRewardOutput(
        motion_reward=motion_reward,
        per_view_reward=per_view_reward,
        valid_mask=valid_mask,
        per_view_valid_mask=per_view_valid_mask,
        diagnostics=diagnostics,
    )


def _invalid_geometry_reward(
    output: GeometryOutput,
    code: str,
    reason: str,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> SingleViewGeometryReward:
    details = dict(diagnostics or {})
    details["invalid_code"] = code
    details["invalid_reason"] = reason
    return SingleViewGeometryReward(
        reward=output.depths.new_tensor(float("nan")),
        valid=False,
        frame_errors=output.depths.new_empty((0,)),
        diagnostics=details,
    )


def _validate_geometry_shapes(output: GeometryOutput) -> Optional[str]:
    if output.depths.ndim != 3:
        return f"depths must have shape [T, H, W], got {tuple(output.depths.shape)}"

    num_frames, height, width = output.depths.shape
    expected_shapes = {
        "camera_poses": (num_frames, 4, 4),
        "point_maps": (num_frames, height, width, 3),
        "valid_mask": (num_frames, height, width),
    }
    tensors = {
        "camera_poses": output.camera_poses,
        "point_maps": output.point_maps,
        "valid_mask": output.valid_mask,
    }
    for name, expected_shape in expected_shapes.items():
        if tuple(tensors[name].shape) != expected_shape:
            return f"{name} must have shape {expected_shape}, got {tuple(tensors[name].shape)}"

    if output.scene_flows is not None:
        expected_flow_shape = (num_frames, height, width, 3)
        if tuple(output.scene_flows.shape) != expected_flow_shape:
            return (
                f"scene_flows must have shape {expected_flow_shape}, "
                f"got {tuple(output.scene_flows.shape)}"
            )

    if output.point_valid_mask is not None:
        expected_point_mask_shape = (num_frames, height, width)
        if tuple(output.point_valid_mask.shape) != expected_point_mask_shape:
            return (
                "point_valid_mask must have shape "
                f"{expected_point_mask_shape}, got "
                f"{tuple(output.point_valid_mask.shape)}"
            )
        if output.point_valid_mask.dtype != torch.bool:
            return (
                "point_valid_mask must be bool, got "
                f"{output.point_valid_mask.dtype}"
            )

    if output.intrinsics is None:
        return "intrinsics are required for geometry reprojection"
    expected_intrinsics_shape = (num_frames, 3, 3)
    if tuple(output.intrinsics.shape) != expected_intrinsics_shape:
        return (
            f"intrinsics must have shape {expected_intrinsics_shape}, "
            f"got {tuple(output.intrinsics.shape)}"
        )
    if output.valid_mask.dtype != torch.bool:
        return f"valid_mask must be bool, got {output.valid_mask.dtype}"

    reference_device = output.depths.device
    named_tensors = {
        "camera_poses": output.camera_poses,
        "point_maps": output.point_maps,
        "valid_mask": output.valid_mask,
        "intrinsics": output.intrinsics,
    }
    if output.scene_flows is not None:
        named_tensors["scene_flows"] = output.scene_flows
    if output.point_valid_mask is not None:
        named_tensors["point_valid_mask"] = output.point_valid_mask
    for name, tensor in named_tensors.items():
        if tensor.device != reference_device:
            return f"{name} must be on device {reference_device}, got {tensor.device}"

    return None


def _find_nonfinite_tensor(output: GeometryOutput) -> Optional[str]:
    named_tensors = {
        "camera_poses": output.camera_poses,
        "depths": output.depths,
        "point_maps": output.point_maps,
        "intrinsics": output.intrinsics,
    }
    if output.scene_flows is not None:
        named_tensors["scene_flows"] = output.scene_flows
    for name, tensor in named_tensors.items():
        if tensor is not None and not torch.isfinite(tensor).all():
            return name
    return None


def _aggregate_static_world_points(
    output: GeometryOutput,
    config: GeometryRewardConfig,
) -> Tuple[Tensor, Dict[str, Any]]:
    # Point maps and scene flow use a persistent reference-frame grid.  Never
    # apply target-frame depth-pixel validity to that grid when an explicit
    # point mask is available.
    static_mask = (
        output.point_valid_mask.clone()
        if output.point_valid_mask is not None
        else output.valid_mask.clone()
    )
    valid_count = int(static_mask.sum().item())

    if output.scene_flows is not None:
        flow_magnitude = torch.linalg.vector_norm(output.scene_flows, dim=-1)
        # Scene-flow point grids track the same reference pixels over time.
        # A point is static only if it remains below the threshold in every
        # frame; otherwise the zero flow at the reference frame would let a
        # moving point leak into the aggregated static cloud.
        reference_static_mask = (
            flow_magnitude <= config.static_flow_threshold
        ).all(dim=0)
        static_mask &= reference_static_mask.unsqueeze(0)

    static_count = int(static_mask.sum().item())
    diagnostics = {
        "valid_geometry_points": valid_count,
        "static_geometry_points": static_count,
        "static_fraction": static_count / max(valid_count, 1),
    }
    return output.point_maps[static_mask], diagnostics


def _reproject_depth(
    world_points: Tensor,
    camera_to_world: Tensor,
    intrinsics: Tensor,
    height: int,
    width: int,
    eps: float,
) -> Tensor:
    """Project world points and retain the nearest depth at each pixel."""

    world_to_camera = torch.linalg.inv(camera_to_world)
    homogeneous_points = torch.cat(
        [world_points, torch.ones_like(world_points[:, :1])], dim=-1
    )
    camera_points = (homogeneous_points @ world_to_camera.transpose(0, 1))[:, :3]
    depths = camera_points[:, 2]

    in_front = depths > eps
    camera_points = camera_points[in_front]
    depths = depths[in_front]
    if depths.numel() == 0:
        return world_points.new_full((height, width), float("inf"))

    image_points = camera_points @ intrinsics.transpose(0, 1)
    pixel_x = torch.round(image_points[:, 0] / image_points[:, 2]).to(torch.long)
    pixel_y = torch.round(image_points[:, 1] / image_points[:, 2]).to(torch.long)
    in_bounds = (
        (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    pixel_x = pixel_x[in_bounds]
    pixel_y = pixel_y[in_bounds]
    depths = depths[in_bounds]

    depth_buffer = world_points.new_full((height * width,), float("inf"))
    if depths.numel() > 0:
        flat_indices = pixel_y * width + pixel_x
        depth_buffer.scatter_reduce_(
            0, flat_indices, depths, reduce="amin", include_self=True
        )
    return depth_buffer.view(height, width)


@torch.no_grad()
def compute_single_view_geometry_reward(
    output: GeometryOutput,
    config: GeometryRewardConfig,
    video: Optional[Tensor] = None,
) -> SingleViewGeometryReward:
    """Compute worst-frame reprojection consistency for one video view.

    For every frame, all valid static world points are projected into that
    frame.  The per-frame error is the mean absolute depth difference on pixels
    where both projected and predicted depth are valid.  The returned reward is
    the negative mean of the worst ``topk_worst_frames`` errors, so larger is
    better as required by the VGGRPO-style objective.
    """

    shape_error = _validate_geometry_shapes(output)
    if shape_error is not None:
        return _invalid_geometry_reward(output, "invalid_geometry_shape", shape_error)

    nonfinite_tensor = _find_nonfinite_tensor(output)
    if nonfinite_tensor is not None:
        return _invalid_geometry_reward(
            output,
            "nonfinite_geometry",
            f"non-finite values in {nonfinite_tensor}",
        )

    num_frames, height, width = output.depths.shape
    if num_frames < config.topk_worst_frames:
        return _invalid_geometry_reward(
            output,
            "insufficient_frames",
            f"only {num_frames} frames for top-{config.topk_worst_frames} aggregation",
        )

    world_points, diagnostics = _aggregate_static_world_points(output, config)
    if world_points.numel() == 0:
        return _invalid_geometry_reward(
            output,
            "no_static_geometry",
            "no valid static geometry points",
            diagnostics,
        )

    frame_errors: List[Tensor] = []
    projected_pixels_per_frame: List[int] = []
    try:
        for frame_index in range(num_frames):
            reprojected_depth = _reproject_depth(
                world_points=world_points,
                camera_to_world=output.camera_poses[frame_index],
                intrinsics=output.intrinsics[frame_index],
                height=height,
                width=width,
                eps=config.eps,
            )
            comparison_mask = output.valid_mask[frame_index] & torch.isfinite(
                reprojected_depth
            )
            projected_count = int(comparison_mask.sum().item())
            projected_pixels_per_frame.append(projected_count)
            if projected_count < config.min_valid_projected_pixels:
                continue

            frame_errors.append(
                torch.abs(
                    reprojected_depth[comparison_mask]
                    - output.depths[frame_index][comparison_mask]
                ).mean()
            )
    except RuntimeError as error:
        diagnostics["projected_pixels_per_frame"] = projected_pixels_per_frame
        return _invalid_geometry_reward(
            output,
            "reprojection_failure",
            f"reprojection failed: {error}",
            diagnostics,
        )

    diagnostics["projected_pixels_per_frame"] = projected_pixels_per_frame
    diagnostics["valid_reprojection_frames"] = len(frame_errors)
    if len(frame_errors) < config.topk_worst_frames:
        return _invalid_geometry_reward(
            output,
            "insufficient_projected_pixels",
            (
                f"only {len(frame_errors)} frames have at least "
                f"{config.min_valid_projected_pixels} projected pixels"
            ),
            diagnostics,
        )

    all_frame_errors = torch.stack(frame_errors)
    worst_errors = torch.topk(
        all_frame_errors, k=config.topk_worst_frames, largest=True
    ).values
    diagnostics["worst_frame_errors"] = worst_errors
    static_error = worst_errors.mean()
    diagnostics["static_reprojection_error"] = static_error

    # Direct formula audits may omit RGB and intentionally exercise only the
    # Any4D reprojection component.  The training path always supplies video.
    if video is None or config.rgb_temporal_weight == 0.0:
        reward = -static_error
        diagnostics["reward_version"] = "static_reprojection_only"
    else:
        if video.shape[0] < config.min_rgb_temporal_frames:
            return _invalid_geometry_reward(
                output,
                "insufficient_rgb_temporal_frames",
                (
                    f"only {video.shape[0]} RGB frames; need at least "
                    f"{config.min_rgb_temporal_frames}"
                ),
                diagnostics,
            )
        try:
            rgb_error, rgb_diagnostics = compute_rgb_temporal_error(
                video,
                resize_long_side=config.rgb_resize_long_side,
                topk_worst_pairs=config.topk_worst_frames,
            )
        except (RuntimeError, ValueError) as error:
            return _invalid_geometry_reward(
                output,
                "rgb_temporal_failure",
                f"RGB temporal consistency failed: {error}",
                diagnostics,
            )
        rgb_error = rgb_error.to(static_error.device)
        static_normalized = static_error / config.static_error_scale
        reward = -(
            (1.0 - config.rgb_temporal_weight) * static_normalized
            + config.rgb_temporal_weight * rgb_error
        )
        diagnostics.update(
            {
                "reward_version": "any4d_rgb_temporal_v2",
                "static_normalized_error": static_normalized,
                "rgb_temporal_error": rgb_error,
                "rgb_temporal": rgb_diagnostics,
            }
        )

    return SingleViewGeometryReward(
        reward=reward,
        valid=True,
        frame_errors=all_frame_errors,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def compute_geometry_rewards(
    videos: Tensor,
    adapter: GeometryAdapter,
    config: GeometryRewardConfig,
) -> GeometryRewardOutput:
    """Compute independent per-view and view-averaged geometry rewards.

    Args:
        videos: RGB videos shaped ``[K, V, T, C, H, W]`` or
            ``[B, K, V, T, C, H, W]``.  Dimensions are never inferred by
            reshaping: ``K`` is GRPO group size, ``V`` is number of views,
            and ``T`` is number of future frames used by the reward.
        adapter: Frozen geometry backend called once per candidate/view with a
            tensor shaped ``[T, C, H, W]``.
        config: Geometry reward and validity thresholds.

    A candidate is valid only when every configured view is valid.  Invalid
    views are never dropped from the equal-view average.
    """

    if videos.ndim == 6:
        batched_videos = videos.unsqueeze(0)
        squeeze_batch = True
    elif videos.ndim == 7:
        batched_videos = videos
        squeeze_batch = False
    else:
        raise ValueError(
            "videos must have shape [K, V, T, C, H, W] or "
            f"[B, K, V, T, C, H, W], got {tuple(videos.shape)}"
        )

    batch_size, group_size, num_views = batched_videos.shape[:3]
    output_shape = (batch_size, group_size)
    per_view_shape = (batch_size, group_size, num_views)
    per_view_reward = torch.full(
        per_view_shape,
        float("nan"),
        dtype=torch.float32,
        device=videos.device,
    )
    per_view_valid_mask = torch.zeros(
        per_view_shape, dtype=torch.bool, device=videos.device
    )
    geometry_reward = torch.full(
        output_shape,
        float("nan"),
        dtype=torch.float32,
        device=videos.device,
    )
    valid_mask = torch.zeros(output_shape, dtype=torch.bool, device=videos.device)
    diagnostics: Dict[str, Any] = {
        "input_shape": tuple(videos.shape),
        "invalid_views": [],
        "per_view": {},
    }

    if num_views != config.num_views:
        diagnostics["invalid_code"] = "view_count_mismatch"
        diagnostics["invalid_reason"] = (
            f"expected {config.num_views} views, got {num_views}"
        )
    else:
        for batch_index in range(batch_size):
            for candidate_index in range(group_size):
                for view_index in range(num_views):
                    diagnostic_key = (
                        f"batch{batch_index}/candidate{candidate_index}/view{view_index}"
                    )
                    video = batched_videos[
                        batch_index, candidate_index, view_index
                    ]
                    try:
                        geometry = adapter.infer(video)
                        view_result = compute_single_view_geometry_reward(
                            geometry, config, video=video
                        )
                    except Exception as error:
                        diagnostics["invalid_views"].append(
                            {
                                "batch_index": batch_index,
                                "candidate_index": candidate_index,
                                "view_index": view_index,
                                "code": "geometry_inference_failure",
                                "reason": (
                                    f"geometry inference failed: "
                                    f"{type(error).__name__}: {error}"
                                ),
                            }
                        )
                        continue

                    diagnostics["per_view"][diagnostic_key] = view_result.diagnostics
                    if not view_result.valid:
                        diagnostics["invalid_views"].append(
                            {
                                "batch_index": batch_index,
                                "candidate_index": candidate_index,
                                "view_index": view_index,
                                "code": view_result.diagnostics.get(
                                    "invalid_code", "invalid_geometry"
                                ),
                                "reason": view_result.diagnostics.get(
                                    "invalid_reason", "unknown geometry error"
                                ),
                            }
                        )
                        continue

                    per_view_reward[
                        batch_index, candidate_index, view_index
                    ] = view_result.reward.to(
                        device=videos.device, dtype=torch.float32
                    )
                    per_view_valid_mask[
                        batch_index, candidate_index, view_index
                    ] = True

                candidate_views_valid = per_view_valid_mask[
                    batch_index, candidate_index
                ].all()
                if bool(candidate_views_valid.item()):
                    geometry_reward[batch_index, candidate_index] = per_view_reward[
                        batch_index, candidate_index
                    ].mean()
                    valid_mask[batch_index, candidate_index] = True

    diagnostics["valid_rate"] = float(valid_mask.float().mean().item())
    diagnostics["per_view_valid_rate"] = [
        float(per_view_valid_mask[..., view_index].float().mean().item())
        for view_index in range(num_views)
    ]
    invalid_reason_counts: Dict[str, int] = {}
    for invalid_view in diagnostics["invalid_views"]:
        code = invalid_view["code"]
        invalid_reason_counts[code] = invalid_reason_counts.get(code, 0) + 1
    if "invalid_code" in diagnostics:
        code = diagnostics["invalid_code"]
        invalid_reason_counts[code] = batch_size * group_size
    diagnostics["invalid_reason_counts"] = invalid_reason_counts

    if squeeze_batch:
        geometry_reward = geometry_reward.squeeze(0)
        per_view_reward = per_view_reward.squeeze(0)
        valid_mask = valid_mask.squeeze(0)
        per_view_valid_mask = per_view_valid_mask.squeeze(0)

    return GeometryRewardOutput(
        geometry_reward=geometry_reward,
        per_view_reward=per_view_reward,
        valid_mask=valid_mask,
        per_view_valid_mask=per_view_valid_mask,
        diagnostics=diagnostics,
    )


class _RecordingGeometryAdapter(GeometryAdapter):
    """Record one traversal so another reward component can replay it."""

    def __init__(self, adapter: GeometryAdapter) -> None:
        self.adapter = adapter
        self.records: List[
            Tuple[Optional[GeometryOutput], Optional[Exception]]
        ] = []

    @torch.no_grad()
    def infer(self, video: Tensor) -> GeometryOutput:
        try:
            output = self.adapter.infer(video)
        except Exception as error:
            self.records.append((None, error))
            raise
        self.records.append((output, None))
        return output


class _ReplayGeometryAdapter(GeometryAdapter):
    """Replay recorded outputs/errors in the same B/K/V traversal order."""

    def __init__(
        self,
        records: List[Tuple[Optional[GeometryOutput], Optional[Exception]]],
    ) -> None:
        self.records = records
        self.next_index = 0

    @torch.no_grad()
    def infer(self, video: Tensor) -> GeometryOutput:
        if self.next_index >= len(self.records):
            raise RuntimeError("geometry inference replay is exhausted")
        output, error = self.records[self.next_index]
        self.next_index += 1
        if error is not None:
            raise error
        if output is None:
            raise RuntimeError("geometry inference replay contains no output")
        return output


@torch.no_grad()
def compute_vggrpo_reward_components(
    videos: Tensor,
    adapter: GeometryAdapter,
    geometry_config: GeometryRewardConfig,
    motion_config: CameraMotionRewardConfig,
) -> VGGRPORewardComponents:
    """Compute motion and geometry while invoking the heavy adapter only once."""

    if geometry_config.num_views != motion_config.num_views:
        raise ValueError(
            "geometry and motion num_views must match, got "
            f"{geometry_config.num_views} and {motion_config.num_views}"
        )
    recording_adapter = _RecordingGeometryAdapter(adapter)
    geometry = compute_geometry_rewards(
        videos=videos,
        adapter=recording_adapter,
        config=geometry_config,
    )
    replay_adapter = _ReplayGeometryAdapter(recording_adapter.records)
    motion = compute_camera_motion_rewards(
        videos=videos,
        adapter=replay_adapter,
        config=motion_config,
    )
    if replay_adapter.next_index != len(recording_adapter.records):
        raise RuntimeError(
            "motion and geometry traversals consumed different inference counts"
        )
    return VGGRPORewardComponents(motion=motion, geometry=geometry)


@torch.no_grad()
def normalize_and_combine_group_rewards(
    motion_reward: Tensor,
    geometry_reward: Tensor,
    valid_mask: Tensor,
    config: GroupNormalizationConfig,
) -> GroupNormalizationOutput:
    """Normalize reward components separately within each GRPO group.

    Inputs have shape ``[K]`` or ``[B, K]``.  A single common validity mask is
    deliberately used for both statistics: a candidate that cannot contribute
    a final advantage must not influence either component's group mean or
    standard deviation.  Non-finite raw rewards are made invalid explicitly.

    Population standard deviation (``unbiased=False``) implements the sigma in
    the task specification.  Raw rewards are never summed before normalization.
    """

    if motion_reward.shape != geometry_reward.shape:
        raise ValueError(
            "motion_reward and geometry_reward must have the same shape, got "
            f"{tuple(motion_reward.shape)} and {tuple(geometry_reward.shape)}"
        )
    if valid_mask.shape != motion_reward.shape:
        raise ValueError(
            "valid_mask must match reward shape, got "
            f"{tuple(valid_mask.shape)} and {tuple(motion_reward.shape)}"
        )
    if motion_reward.ndim not in (1, 2):
        raise ValueError(
            "group rewards must have shape [K] or [B, K], got "
            f"{tuple(motion_reward.shape)}"
        )
    if valid_mask.dtype != torch.bool:
        raise ValueError(f"valid_mask must be bool, got {valid_mask.dtype}")
    if not motion_reward.is_floating_point() or not geometry_reward.is_floating_point():
        raise ValueError("motion_reward and geometry_reward must be floating point")
    if motion_reward.device != geometry_reward.device or valid_mask.device != motion_reward.device:
        raise ValueError("rewards and valid_mask must be on the same device")

    if motion_reward.ndim == 1:
        batched_motion = motion_reward.unsqueeze(0)
        batched_geometry = geometry_reward.unsqueeze(0)
        batched_valid = valid_mask.unsqueeze(0)
        squeeze_batch = True
    else:
        batched_motion = motion_reward
        batched_geometry = geometry_reward
        batched_valid = valid_mask
        squeeze_batch = False

    finite_rewards = torch.isfinite(batched_motion) & torch.isfinite(
        batched_geometry
    )
    effective_valid = batched_valid & finite_rewards
    normalized_motion = torch.full_like(batched_motion, float("nan"))
    normalized_geometry = torch.full_like(batched_geometry, float("nan"))
    advantage = torch.full_like(batched_motion, float("nan"))
    valid_group_mask = torch.zeros(
        batched_motion.shape[0], dtype=torch.bool, device=motion_reward.device
    )
    group_diagnostics: List[Dict[str, Any]] = []

    for batch_index in range(batched_motion.shape[0]):
        group_valid = effective_valid[batch_index]
        valid_count = int(group_valid.sum().item())
        details: Dict[str, Any] = {"valid_candidate_count": valid_count}
        if valid_count < config.min_valid_group_size:
            details["invalid_reason"] = (
                f"only {valid_count} valid candidates; "
                f"need at least {config.min_valid_group_size}"
            )
            group_diagnostics.append(details)
            continue

        valid_motion = batched_motion[batch_index][group_valid]
        valid_geometry = batched_geometry[batch_index][group_valid]
        motion_mean = valid_motion.mean()
        motion_std = valid_motion.std(unbiased=False)
        geometry_mean = valid_geometry.mean()
        geometry_std = valid_geometry.std(unbiased=False)

        group_normalized_motion = (valid_motion - motion_mean) / (
            motion_std + config.eps
        )
        group_normalized_geometry = (valid_geometry - geometry_mean) / (
            geometry_std + config.eps
        )
        group_advantage = (
            config.motion_weight * group_normalized_motion
            + config.geometry_weight * group_normalized_geometry
        )

        normalized_motion[batch_index][group_valid] = group_normalized_motion
        normalized_geometry[batch_index][group_valid] = group_normalized_geometry
        advantage[batch_index][group_valid] = group_advantage
        valid_group_mask[batch_index] = True
        details.update(
            {
                "motion_mean": float(motion_mean.item()),
                "motion_std": float(motion_std.item()),
                "geometry_mean": float(geometry_mean.item()),
                "geometry_std": float(geometry_std.item()),
                "advantage_mean": float(group_advantage.mean().item()),
                "advantage_std": float(
                    group_advantage.std(unbiased=False).item()
                ),
            }
        )
        group_diagnostics.append(details)

    effective_valid &= valid_group_mask.unsqueeze(-1)
    diagnostics = {
        "groups": group_diagnostics,
        "valid_group_rate": float(valid_group_mask.float().mean().item()),
        "excluded_nonfinite_count": int(
            (batched_valid & ~finite_rewards).sum().item()
        ),
    }

    if squeeze_batch:
        normalized_motion = normalized_motion.squeeze(0)
        normalized_geometry = normalized_geometry.squeeze(0)
        advantage = advantage.squeeze(0)
        effective_valid = effective_valid.squeeze(0)
        valid_group_mask = valid_group_mask.squeeze(0)

    return GroupNormalizationOutput(
        normalized_motion_reward=normalized_motion,
        normalized_geometry_reward=normalized_geometry,
        advantage=advantage,
        valid_mask=effective_valid,
        valid_group_mask=valid_group_mask,
        diagnostics=diagnostics,
    )


def _masked_mean_std(values: Tensor, valid_mask: Tensor) -> Tuple[float, float]:
    selected = values[valid_mask & torch.isfinite(values)]
    if selected.numel() == 0:
        return float("nan"), float("nan")
    return (
        float(selected.mean().item()),
        float(selected.std(unbiased=False).item()),
    )


@torch.no_grad()
def assemble_reward_output(
    motion_reward: Tensor,
    motion_valid_mask: Tensor,
    geometry_output: GeometryRewardOutput,
    config: GroupNormalizationConfig,
    motion_output: Optional[CameraMotionRewardOutput] = None,
) -> RewardOutput:
    """Combine component validity and build the training-facing output.

    A candidate participates in normalization only when both components are
    valid and finite.  If every group is invalid, ``advantage`` is returned as
    ``None`` rather than as a usable-looking tensor of placeholder values.
    """

    if motion_reward.shape != geometry_output.geometry_reward.shape:
        raise ValueError(
            "motion_reward and geometry_reward must have the same group layout, "
            f"got {tuple(motion_reward.shape)} and "
            f"{tuple(geometry_output.geometry_reward.shape)}"
        )
    if motion_valid_mask.shape != motion_reward.shape:
        raise ValueError("motion_valid_mask must match motion_reward shape")
    if motion_valid_mask.dtype != torch.bool:
        raise ValueError("motion_valid_mask must be bool")

    component_valid_mask = motion_valid_mask & geometry_output.valid_mask
    normalized = normalize_and_combine_group_rewards(
        motion_reward=motion_reward,
        geometry_reward=geometry_output.geometry_reward,
        valid_mask=component_valid_mask,
        config=config,
    )

    motion_mean, motion_std = _masked_mean_std(
        motion_reward, component_valid_mask
    )
    geometry_mean, geometry_std = _masked_mean_std(
        geometry_output.geometry_reward, component_valid_mask
    )
    if normalized.valid_mask.any():
        advantage_mean, advantage_std = _masked_mean_std(
            normalized.advantage, normalized.valid_mask
        )
        advantage: Optional[Tensor] = normalized.advantage
    else:
        advantage_mean, advantage_std = float("nan"), float("nan")
        advantage = None

    diagnostics: Dict[str, Any] = {
        "reward/motion_mean": motion_mean,
        "reward/motion_std": motion_std,
        "reward/geo_mean": geometry_mean,
        "reward/geo_std": geometry_std,
        "reward/advantage_mean": advantage_mean,
        "reward/advantage_std": advantage_std,
        "reward/valid_rate": float(normalized.valid_mask.float().mean().item()),
        "raw_component_valid_rate": float(
            component_valid_mask.float().mean().item()
        ),
        "geometry": geometry_output.diagnostics,
        "motion": (
            motion_output.diagnostics if motion_output is not None else {}
        ),
        "normalization": normalized.diagnostics,
    }
    num_views = geometry_output.per_view_reward.shape[-1]
    for view_index in range(num_views):
        view_mean, _ = _masked_mean_std(
            geometry_output.per_view_reward[..., view_index],
            geometry_output.per_view_valid_mask[..., view_index],
        )
        diagnostics[f"reward/view{view_index}_geo"] = view_mean
        if motion_output is not None:
            view_motion_mean, _ = _masked_mean_std(
                motion_output.per_view_reward[..., view_index],
                motion_output.per_view_valid_mask[..., view_index],
            )
            diagnostics[f"reward/view{view_index}_motion"] = view_motion_mean

    return RewardOutput(
        motion_reward=motion_reward,
        geometry_reward=geometry_output.geometry_reward,
        advantage=advantage,
        valid_mask=normalized.valid_mask,
        valid_group_mask=normalized.valid_group_mask,
        normalized_motion_reward=normalized.normalized_motion_reward,
        normalized_geometry_reward=normalized.normalized_geometry_reward,
        per_view_motion_reward=(
            motion_output.per_view_reward if motion_output is not None else None
        ),
        per_view_motion_valid_mask=(
            motion_output.per_view_valid_mask
            if motion_output is not None
            else None
        ),
        per_view_geometry_reward=geometry_output.per_view_reward,
        per_view_valid_mask=geometry_output.per_view_valid_mask,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def compute_vggrpo_reward(
    videos: Tensor,
    adapter: GeometryAdapter,
    geometry_config: GeometryRewardConfig,
    motion_config: CameraMotionRewardConfig,
    normalization_config: GroupNormalizationConfig,
) -> RewardOutput:
    """Training-facing full VGGRPO-style RGB reward entry point."""

    components = compute_vggrpo_reward_components(
        videos=videos,
        adapter=adapter,
        geometry_config=geometry_config,
        motion_config=motion_config,
    )
    return assemble_reward_output(
        motion_reward=components.motion.motion_reward,
        motion_valid_mask=components.motion.valid_mask,
        geometry_output=components.geometry,
        config=normalization_config,
        motion_output=components.motion,
    )
