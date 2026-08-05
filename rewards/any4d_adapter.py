"""Any4D RGB backend for the VGGRPO-style geometry reward.

Any4D is an optional external dependency.  This module never downloads model
weights and imports Any4D lazily, so mock reward tests stay lightweight.  The
official model can be loaded from an explicit local config/checkpoint with
``load_official_any4d_model`` and passed to ``Any4DGeometryAdapter``.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .geometry_adapter import GeometryAdapter, GeometryOutput


InferenceFunction = Callable[..., Dict[str, Any]]
QuaternionToMatrixFunction = Callable[[Tensor], Tensor]
RecoverIntrinsicsFunction = Callable[[Tensor], Tensor]


@contextmanager
def _construct_dinov2_without_pretrained_download() -> Iterator[None]:
    """Build the DINOv2 architecture without fetching its standalone weights.

    UniCeption's DINOv2 constructor always calls ``torch.hub.load`` with the
    upstream default ``pretrained=True``.  A complete Any4D checkpoint contains
    the encoder state and is loaded immediately after model construction, so
    downloading the standalone 1.13 GB DINOv2 state is redundant.  Restrict the
    override to the official DINOv2 hub call and restore ``torch.hub.load`` as
    soon as construction finishes.
    """

    original_hub_load = torch.hub.load

    def load_architecture_only(
        repo_or_dir: str, model_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        if repo_or_dir == "facebookresearch/dinov2":
            kwargs["pretrained"] = False
        return original_hub_load(repo_or_dir, model_name, *args, **kwargs)

    torch.hub.load = load_architecture_only
    try:
        yield
    finally:
        torch.hub.load = original_hub_load


def load_official_any4d_model(
    config_path: Path,
    checkpoint_path: Path,
    config_overrides: Optional[Sequence[str]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the official Any4D model from explicit local files.

    This function performs no download and initially keeps the model on CPU.
    ``Any4DGeometryAdapter`` subsequently freezes it and moves it to the chosen
    device.  The checkpoint is loaded with ``weights_only=False`` because the
    official file contains OmegaConf metadata in addition to tensors; callers
    must therefore use a checkpoint obtained from the trusted official source.

    Args:
        config_path: Official Any4D ``configs/train.yaml`` path.
        checkpoint_path: Official local ``any4d_4v_combined.pth`` path.
        config_overrides: Optional Hydra overrides.  Defaults to the official
            RGB-only inference configuration.

    Returns:
        The initialized model and state-dict loading diagnostics.
    """

    resolved_config = Path(config_path).expanduser().resolve()
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not resolved_config.is_file():
        raise FileNotFoundError(f"Any4D config not found: {resolved_config}")
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(
            f"Any4D checkpoint not found: {resolved_checkpoint}"
        )

    try:
        import hydra
        from hydra.core.global_hydra import GlobalHydra

        from any4d.models import init_model
    except ImportError as error:
        raise ImportError(
            "Official Any4D model loading requires hydra-core and the Any4D "
            "package in the active environment."
        ) from error

    overrides = list(
        config_overrides
        or (
            "machine=local",
            "model=any4d",
            "model.encoder.uses_torch_hub=false",
            "model/task=images_only",
        )
    )
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(resolved_config.parent),
    ):
        cfg = hydra.compose(
            config_name=resolved_config.stem,
            overrides=overrides,
        )

    with _construct_dinov2_without_pretrained_download():
        model = init_model(cfg.model.model_str, cfg.model.model_config)

    checkpoint = torch.load(
        resolved_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            "Official Any4D checkpoint must contain a 'model' state dict"
        )
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    diagnostics = {
        "config_path": str(resolved_config),
        "checkpoint_path": str(resolved_checkpoint),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "skipped_standalone_dinov2_download": True,
    }
    return model, diagnostics


@dataclass(frozen=True)
class Any4DAdapterConfig:
    """Tensor preprocessing and inference settings for Any4D."""

    longest_side: int = 518
    patch_size: int = 14
    data_norm_type: str = "dinov2"
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    input_range: str = "zero_one"
    use_amp: bool = True
    amp_dtype: str = "bf16"
    require_scene_flow: bool = True

    def __post_init__(self) -> None:
        if self.longest_side <= 0:
            raise ValueError("longest_side must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.longest_side % self.patch_size != 0:
            raise ValueError("longest_side must be divisible by patch_size")
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("image_mean and image_std must contain three values")
        if any(std <= 0 for std in self.image_std):
            raise ValueError("image_std values must be positive")
        if self.input_range not in ("zero_one", "minus_one_one"):
            raise ValueError("input_range must be 'zero_one' or 'minus_one_one'")
        if self.amp_dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError("amp_dtype must be fp32, fp16, or bf16")


def _load_any4d_functions() -> Tuple[
    InferenceFunction,
    QuaternionToMatrixFunction,
    RecoverIntrinsicsFunction,
]:
    try:
        from any4d.utils.geometry import (
            quaternion_to_rotation_matrix,
            recover_pinhole_intrinsics_from_ray_directions,
        )
        from any4d.utils.inference import loss_of_one_batch_multi_view
    except ImportError as error:
        raise ImportError(
            "Any4D is not installed. Install the official Any-4D/Any4D package "
            "in a compatible environment and initialize its checkpoint before "
            "constructing Any4DGeometryAdapter."
        ) from error
    return (
        loss_of_one_batch_multi_view,
        quaternion_to_rotation_matrix,
        recover_pinhole_intrinsics_from_ray_directions,
    )


def _remove_singleton_batch(tensor: Tensor, name: str) -> Tensor:
    if tensor.ndim == 0 or tensor.shape[0] != 1:
        raise ValueError(
            f"Any4D output {name} must start with batch size 1, "
            f"got {tuple(tensor.shape)}"
        )
    return tensor[0].float()


class Any4DGeometryAdapter(GeometryAdapter):
    """Run a frozen official Any4D model on one RGB video/view.

    The official checkpoint predicts target-frame scene flow on the reference
    frame's point grid.  This adapter therefore constructs frame ``t`` point
    maps as ``reference_points + flow_t``.  That preserves correspondence
    between ``point_maps`` and ``scene_flows`` and lets the reward reject a
    point as dynamic if it moves at any time.

    Model construction/checkpoint loading stays outside this adapter because
    the official repository uses Hydra configs and is still evolving.  This
    class owns preprocessing, frozen inference, and output conversion only.
    """

    def __init__(
        self,
        model: Any,
        device: torch.device,
        config: Optional[Any4DAdapterConfig] = None,
        inference_fn: Optional[InferenceFunction] = None,
        quaternion_to_matrix_fn: Optional[QuaternionToMatrixFunction] = None,
        recover_intrinsics_fn: Optional[RecoverIntrinsicsFunction] = None,
    ) -> None:
        self.config = config or Any4DAdapterConfig()
        self.device = torch.device(device)
        if (
            inference_fn is None
            or quaternion_to_matrix_fn is None
            or recover_intrinsics_fn is None
        ):
            official_functions = _load_any4d_functions()
            inference_fn = inference_fn or official_functions[0]
            quaternion_to_matrix_fn = (
                quaternion_to_matrix_fn or official_functions[1]
            )
            recover_intrinsics_fn = recover_intrinsics_fn or official_functions[2]

        self._inference_fn = inference_fn
        self._quaternion_to_matrix_fn = quaternion_to_matrix_fn
        self._recover_intrinsics_fn = recover_intrinsics_fn
        self.model = model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _target_size(self, height: int, width: int) -> Tuple[int, int]:
        if width >= height:
            target_width = self.config.longest_side
            target_height = round(
                self.config.longest_side / (width / height) / self.config.patch_size
            ) * self.config.patch_size
        else:
            target_height = self.config.longest_side
            target_width = round(
                self.config.longest_side * (width / height) / self.config.patch_size
            ) * self.config.patch_size
        return (
            max(self.config.patch_size, target_height),
            max(self.config.patch_size, target_width),
        )

    def _prepare_views(self, video: Tensor) -> List[Dict[str, Any]]:
        if video.ndim != 4:
            raise ValueError(
                f"Any4D expects video [T, C, H, W], got {tuple(video.shape)}"
            )
        if video.shape[1] != 3:
            raise ValueError(f"Any4D expects RGB input, got C={video.shape[1]}")
        if not video.is_floating_point():
            raise ValueError("Any4D video input must be floating point")
        if not torch.isfinite(video).all():
            raise ValueError("Any4D video input contains NaN or Inf")

        rgb = video.to(device=self.device, dtype=torch.float32)
        if self.config.input_range == "minus_one_one":
            rgb = (rgb + 1.0) / 2.0
        if (rgb < 0).any() or (rgb > 1).any():
            raise ValueError(
                f"Any4D {self.config.input_range} input falls outside its declared range"
            )

        target_height, target_width = self._target_size(
            height=rgb.shape[-2], width=rgb.shape[-1]
        )
        resized = F.interpolate(
            rgb,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        mean = resized.new_tensor(self.config.image_mean).view(1, 3, 1, 1)
        std = resized.new_tensor(self.config.image_std).view(1, 3, 1, 1)
        normalized = (resized - mean) / std

        views: List[Dict[str, Any]] = []
        for frame_index in range(normalized.shape[0]):
            views.append(
                {
                    "img": normalized[frame_index : frame_index + 1],
                    "true_shape": np.int32([[target_height, target_width]]),
                    "idx": frame_index,
                    "instance": str(frame_index),
                    "data_norm_type": [self.config.data_norm_type],
                }
            )
        return views

    def _convert_predictions(
        self,
        predictions: Dict[str, Any],
        num_frames: int,
        input_shape: Tuple[int, ...],
    ) -> GeometryOutput:
        per_frame_predictions: List[Dict[str, Tensor]] = []
        for frame_index in range(num_frames):
            key = f"pred{frame_index + 1}"
            if key not in predictions:
                raise ValueError(f"Any4D inference result is missing {key}")
            per_frame_predictions.append(predictions[key])

        reference_points = _remove_singleton_batch(
            per_frame_predictions[0]["pts3d"], "pred1.pts3d"
        )
        zero_flow = torch.zeros_like(reference_points)
        scene_flows: List[Tensor] = [zero_flow]
        for frame_index, prediction in enumerate(per_frame_predictions[1:], start=1):
            if "scene_flow" not in prediction:
                if self.config.require_scene_flow:
                    raise ValueError(
                        f"Any4D pred{frame_index + 1} is missing scene_flow"
                    )
                scene_flows = []
                break
            scene_flows.append(
                _remove_singleton_batch(
                    prediction["scene_flow"],
                    f"pred{frame_index + 1}.scene_flow",
                )
            )

        if scene_flows:
            stacked_flows: Optional[Tensor] = torch.stack(scene_flows)
            point_maps = reference_points.unsqueeze(0) + stacked_flows
        else:
            stacked_flows = None
            point_maps = torch.stack(
                [
                    _remove_singleton_batch(
                        prediction["pts3d"], f"pred{index + 1}.pts3d"
                    )
                    for index, prediction in enumerate(per_frame_predictions)
                ]
            )

        camera_poses: List[Tensor] = []
        depths: List[Tensor] = []
        intrinsics: List[Tensor] = []
        confidences: List[Tensor] = []
        for frame_index, prediction in enumerate(per_frame_predictions):
            quaternion = _remove_singleton_batch(
                prediction["cam_quats"], f"pred{frame_index + 1}.cam_quats"
            )
            translation = _remove_singleton_batch(
                prediction["cam_trans"], f"pred{frame_index + 1}.cam_trans"
            )
            rotation = self._quaternion_to_matrix_fn(quaternion).float()
            camera_pose = torch.eye(4, device=rotation.device, dtype=torch.float32)
            camera_pose[:3, :3] = rotation
            camera_pose[:3, 3] = translation
            camera_poses.append(camera_pose)

            if "pts3d_cam" in prediction:
                camera_points = _remove_singleton_batch(
                    prediction["pts3d_cam"],
                    f"pred{frame_index + 1}.pts3d_cam",
                )
            else:
                ray_directions = _remove_singleton_batch(
                    prediction["ray_directions"],
                    f"pred{frame_index + 1}.ray_directions",
                )
                ray_depth = _remove_singleton_batch(
                    prediction["depth_along_ray"],
                    f"pred{frame_index + 1}.depth_along_ray",
                )
                if ray_depth.ndim == 2:
                    ray_depth = ray_depth.unsqueeze(-1)
                camera_points = ray_directions * ray_depth
            depths.append(camera_points[..., 2])

            ray_directions = _remove_singleton_batch(
                prediction["ray_directions"],
                f"pred{frame_index + 1}.ray_directions",
            )
            intrinsics.append(
                self._recover_intrinsics_fn(ray_directions).float()
            )
            if "conf" in prediction:
                confidences.append(
                    _remove_singleton_batch(
                        prediction["conf"], f"pred{frame_index + 1}.conf"
                    )
                )

        stacked_depths = torch.stack(depths)
        # Any4D scene flow and the derived point maps live on frame 0's
        # persistent pixel grid.  Predicted depths live on each target frame's
        # own image grid.  Keep their masks separate even though both tensors
        # happen to have shape [T, H, W].
        point_valid_mask = torch.isfinite(point_maps).all(dim=-1)
        depth_valid_mask = torch.isfinite(stacked_depths) & (stacked_depths > 0)

        metric_scales = []
        for prediction in per_frame_predictions:
            scale = prediction.get("metric_scaling_factor")
            if scale is not None:
                metric_scales.append(float(scale.detach().float().reshape(-1)[0]))
        return GeometryOutput(
            camera_poses=torch.stack(camera_poses),
            depths=stacked_depths,
            point_maps=point_maps,
            scene_flows=stacked_flows,
            valid_mask=depth_valid_mask,
            intrinsics=torch.stack(intrinsics),
            point_valid_mask=point_valid_mask,
            confidence=(
                torch.stack(confidences)
                if len(confidences) == num_frames
                else None
            ),
            diagnostics={
                "backend": "any4d_rgb_4d_model",
                "input_shape": input_shape,
                "geometry_shape": tuple(stacked_depths.shape),
                "scene_flow_alignment": "persistent reference-frame point grid",
                "use_amp": self.config.use_amp,
                "amp_dtype": self.config.amp_dtype,
                "metric_scaling_factors": metric_scales,
                "scene_flow_scale_source": "already scaled by Any4D output head",
                "scene_flow_grid": "persistent reference-frame pixel grid",
            },
        )

    @torch.no_grad()
    def infer(self, video: Tensor) -> GeometryOutput:
        views = self._prepare_views(video)
        predictions = self._inference_fn(
            views,
            self.model,
            None,
            self.device,
            use_amp=self.config.use_amp,
            amp_dtype=self.config.amp_dtype,
        )
        return self._convert_predictions(
            predictions=predictions,
            num_frames=video.shape[0],
            input_shape=tuple(video.shape),
        )
