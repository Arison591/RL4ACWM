"""DA3 Mono reward adapter for TempFlow-generated three-view videos."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from da3_reward import (
    DA3ModelRunner,
    DA3RewardConfig,
    compute_da3_gt_reward,
    make_gt_motion_mask,
)


def _read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video has no decoded frames: {path}")
    return frames


def _future_frames(path: Path, start: int, end: int) -> tuple[list[np.ndarray], np.ndarray]:
    frames = _read_video(path)
    if start < 1 or end < start or len(frames) <= end:
        raise ValueError(
            f"expected frames 0..{end} with 1 <= future_start <= future_end, "
            f"got {len(frames)} frames: {path}"
        )
    return frames[start : end + 1], frames[start - 1]


def _resize_mask(mask: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    if tuple(mask.shape[-2:]) == tuple(spatial_shape):
        return mask.bool()
    views, frames = mask.shape[:2]
    resized = F.interpolate(
        mask.float().reshape(views * frames, 1, *mask.shape[-2:]),
        size=tuple(spatial_shape),
        mode="nearest",
    )
    return resized.reshape(views, frames, *spatial_shape).bool()


class DA3MonoVideoReward:
    """Score matching GT/generated clips with independent per-frame DA3 inference.

    The data contract intentionally matches the existing PSNR reward: the same
    camera video templates, generated ``*_color.mp4`` files and future frame
    interval are used.  Only the scoring representation changes from RGB PSNR
    to frozen DA3 pseudo-depth.
    """

    def __init__(self, config: dict[str, Any], *, runner: DA3ModelRunner | None = None) -> None:
        self.config = dict(config)
        self.cameras = tuple(
            self.config.get("geometry_cameras", ("head", "hand_left", "hand_right"))
        )
        if not self.cameras:
            raise ValueError("DA3 Mono reward needs at least one camera")
        self.future_start = int(self.config.get("geometry_future_start", 4))
        self.future_end = int(self.config.get("geometry_future_end", 28))
        if self.future_start < 1 or self.future_end < self.future_start:
            raise ValueError("DA3 Mono future range must satisfy 1 <= start <= end")
        self.score_config = DA3RewardConfig(
            representation=self.config.get("da3_representation", "inverse"),
            alignment=self.config.get("da3_alignment", "affine"),
            alignment_scope=self.config.get("da3_alignment_scope", "view_clip"),
            loss=self.config.get("da3_loss", "huber"),
            trim_fraction=float(self.config.get("da3_trim_fraction", 0.05)),
            huber_delta_scale=float(self.config.get("da3_huber_delta_scale", 0.10)),
            confidence_mode=self.config.get("da3_confidence_mode", "soft"),
            border_fraction=float(self.config.get("da3_border_fraction", 0.02)),
            dynamic_weight=float(self.config.get("da3_dynamic_weight", 0.35)),
            max_alignment_samples=int(
                self.config.get("da3_max_alignment_samples", 100_000)
            ),
            max_reduction_samples=int(
                self.config.get("da3_max_reduction_samples", 16_384)
            ),
        )
        self._runner = runner
        self._gt_cache: dict[str, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]] = {}

    @property
    def runner(self) -> DA3ModelRunner:
        if self._runner is None:
            self._runner = DA3ModelRunner(
                model_id=self.config.get(
                    "da3_model_path",
                    self.config.get("da3_model_id", "depth-anything/DA3-BASE"),
                ),
                device=self.config.get("da3_device", "cuda"),
                process_res=int(self.config.get("da3_process_res", 336)),
                process_res_method=self.config.get(
                    "da3_process_res_method", "upper_bound_resize"
                ),
                source_root=self.config["da3_source_root"],
            )
        return self._runner

    def _infer_views(
        self, images_by_view: Sequence[Sequence[np.ndarray]]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        depths: list[torch.Tensor] = []
        confidences: list[torch.Tensor | None] = []
        for images in images_by_view:
            depth, confidence = self.runner.infer_mono(images)
            depths.append(depth)
            confidences.append(confidence)
        confidence_tensor = (
            torch.stack([value for value in confidences if value is not None])
            if all(value is not None for value in confidences)
            else None
        )
        return torch.stack(depths), confidence_tensor

    def load_model_for_preflight(self) -> dict[str, Any]:
        """Load the frozen model now so preflight catches source/weight/OOM failures."""

        self.runner._load()
        return self.runner.metadata()

    def _gt_inputs(
        self, condition_id: str
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        cached = self._gt_cache.get(condition_id)
        if cached is not None:
            return cached
        templates = self.config["gt_video_templates"]
        gt_images: list[list[np.ndarray]] = []
        observed_images: list[np.ndarray] = []
        for camera in self.cameras:
            path = Path(
                templates[camera].format(condition_id=condition_id, camera=camera)
            )
            future, observed = _future_frames(path, self.future_start, self.future_end)
            gt_images.append(future)
            observed_images.append(observed)
        gt_depth, gt_confidence = self._infer_views(gt_images)
        gt_rgb = torch.stack(
            [
                torch.stack(
                    [torch.from_numpy(frame.copy()).permute(2, 0, 1) for frame in view]
                )
                for view in gt_images
            ]
        ).float().div(255.0)
        observed_rgb = torch.stack(
            [torch.from_numpy(frame.copy()).permute(2, 0, 1) for frame in observed_images]
        ).float().div(255.0)
        dynamic_mask = _resize_mask(
            make_gt_motion_mask(gt_rgb, observed_rgb), gt_depth.shape[-2:]
        )
        cached = (gt_depth, gt_confidence, dynamic_mask)
        self._gt_cache[condition_id] = cached
        return cached

    @torch.no_grad()
    def score_paths(self, *, condition_id: str, prediction_dir: str | Path) -> dict[str, Any]:
        prediction_dir = Path(prediction_dir)
        generated_images = [
            _future_frames(
                prediction_dir / f"{camera}_color.mp4",
                self.future_start,
                self.future_end,
            )[0]
            for camera in self.cameras
        ]
        gt_depth, gt_confidence, dynamic_mask = self._gt_inputs(condition_id)
        generated_depth, generated_confidence = self._infer_views(generated_images)
        result = compute_da3_gt_reward(
            gt_depth,
            generated_depth,
            gt_confidence,
            generated_confidence,
            dynamic_mask,
            self.score_config,
        )
        if not math.isfinite(result.reward):
            raise FloatingPointError("DA3 Mono reward is non-finite")
        details = result.as_dict()
        metrics = {
            "type": "da3_mono_pseudo_depth",
            "error": result.error,
            "full_error": result.full_error,
            "dynamic_error": result.dynamic_error,
            "per_view_error": dict(zip(self.cameras, result.per_view_error.tolist())),
            "per_view_full_error": dict(
                zip(self.cameras, result.per_view_full_error.tolist())
            ),
            "per_view_dynamic_error": dict(
                zip(self.cameras, result.per_view_dynamic_error.tolist())
            ),
            "future_start": self.future_start,
            "future_end": self.future_end,
            "frame_count_per_view": self.future_end - self.future_start + 1,
            "inference_mode": "mono_per_frame_independent",
            "view_aggregation": "mean",
            "reward_formula": (
                f"-({1.0 - self.score_config.dynamic_weight:.6g} * full_error + "
                f"{self.score_config.dynamic_weight:.6g} * dynamic_error), averaged over views"
            ),
        }
        return {
            "valid": True,
            "total_reward": float(result.reward),
            "action_reward": None,
            "geometry_reward": float(result.reward),
            "geometry": {
                "enabled": True,
                "reward": float(result.reward),
                "metrics": metrics,
                "camera_names": list(self.cameras),
                "reason": None,
            },
            "da3_mono": {
                "reward": float(result.reward),
                "details": details,
                "score_config": self.score_config.__dict__,
                "model": self.runner.metadata(),
            },
        }
