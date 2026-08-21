from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from experiments.awm_coca.reward_runner import compute_head_reward
from experiments.tempflow_video.schemas import BranchRollout, OrdinaryRollout


def _read_video(path: str | Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video has no decoded frames: {path}")
    return frames


def _frame_sobel_error(
    gt: np.ndarray, pred: np.ndarray, *, charbonnier_epsilon: float
) -> float:
    if gt.shape[:2] != pred.shape[:2]:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)
    gt_luma = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    pred_luma = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    errors = []
    for dx, dy in ((1, 0), (0, 1)):
        # A 3x3 Sobel step edge has magnitude four. Scaling by 1/4 keeps
        # gradients in interpretable normalized-luma units.
        gt_grad = cv2.Sobel(gt_luma, cv2.CV_32F, dx, dy, ksize=3, scale=0.25)
        pred_grad = cv2.Sobel(pred_luma, cv2.CV_32F, dx, dy, ksize=3, scale=0.25)
        delta = pred_grad - gt_grad
        robust = np.sqrt(delta * delta + charbonnier_epsilon**2) - charbonnier_epsilon
        errors.append(float(np.mean(robust)))
    return float(np.mean(errors))


def compute_multiview_sobel_reward(
    all_camera_videos: dict[str, dict[str, str]],
    *,
    cameras: list[str],
    future_start: int,
    future_end: int,
    mean_weight: float,
    worst_weight: float,
    charbonnier_epsilon: float = 1.0e-3,
) -> dict[str, Any]:
    """Return a GT-aligned edge score; higher (less negative) is better."""
    if future_start < 0 or future_end < future_start:
        raise ValueError("Sobel frame range must satisfy 0 <= start <= end")
    if not math.isfinite(charbonnier_epsilon) or charbonnier_epsilon <= 0.0:
        raise ValueError("Sobel Charbonnier epsilon must be finite and positive")
    weight_sum = float(mean_weight) + float(worst_weight)
    if mean_weight < 0.0 or worst_weight < 0.0 or weight_sum <= 0.0:
        raise ValueError("Sobel aggregation weights must be non-negative and not both zero")

    per_view_error: dict[str, float] = {}
    per_frame_error: dict[str, list[float]] = {}
    required = future_end + 1
    for camera in cameras:
        if camera not in all_camera_videos:
            raise ValueError(f"missing Sobel camera videos: {camera}")
        pair = all_camera_videos[camera]
        gt_frames = _read_video(pair["gt"])
        pred_frames = _read_video(pair["pred"])
        if len(gt_frames) < required or len(pred_frames) < required:
            raise ValueError(
                f"expected at least {required} {camera} frames, "
                f"got gt={len(gt_frames)} pred={len(pred_frames)}"
            )
        frame_errors = [
            _frame_sobel_error(gt, pred, charbonnier_epsilon=charbonnier_epsilon)
            for gt, pred in zip(
                gt_frames[future_start : future_end + 1],
                pred_frames[future_start : future_end + 1],
                strict=True,
            )
        ]
        per_frame_error[camera] = frame_errors
        per_view_error[camera] = float(np.mean(frame_errors))

    normalized_mean_weight = float(mean_weight / weight_sum)
    normalized_worst_weight = float(worst_weight / weight_sum)
    mean_error = float(np.mean(list(per_view_error.values())))
    worst_view_error = float(max(per_view_error.values()))
    balanced_error = (
        normalized_mean_weight * mean_error
        + normalized_worst_weight * worst_view_error
    )
    return {
        "enabled": True,
        "score": -float(balanced_error),
        "balanced_error": float(balanced_error),
        "mean_error": mean_error,
        "worst_view_error": worst_view_error,
        "per_view_error": per_view_error,
        "per_frame_error": per_frame_error,
        "mean_weight": normalized_mean_weight,
        "worst_weight": normalized_worst_weight,
        "future_start": int(future_start),
        "future_end": int(future_end),
        "charbonnier_epsilon": float(charbonnier_epsilon),
        "definition": "negative GT-aligned signed Sobel-x/y Charbonnier error on luma",
    }


def canonical_reward_config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class VideoRewardAdapter:
    """Thin, no-grad adapter around the legacy terminal video reward entry point."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        protocol = self.config.get(
            "time_alignment_protocol", "legacy_frame_index_truncate"
        )
        if protocol != "legacy_frame_index_truncate":
            raise ValueError(
                "the unchanged legacy reward only supports "
                f"time_alignment_protocol=legacy_frame_index_truncate, got {protocol!r}"
            )
        self.config_sha256 = canonical_reward_config_sha256(self.config)

    def legacy_kwargs(
        self,
        *,
        condition_id: str,
        prep_dir: str,
        prediction_dir: str | Path,
    ) -> tuple[str, str, dict[str, Any]]:
        prediction_dir = Path(prediction_dir)
        template = self.config["gt_video_template"]
        gt_path = template.format(condition_id=condition_id)
        cameras = list(self.config.get("geometry_cameras", ("head", "hand_left", "hand_right")))
        templates = self.config.get("gt_video_templates", {})
        all_camera_videos = {
            camera: {
                "gt": templates.get(camera, template).format(
                    condition_id=condition_id, camera=camera
                ),
                "pred": str(prediction_dir / f"{camera}_color.mp4"),
            }
            for camera in cameras
        }
        kwargs = {
            "prep_dir": prep_dir,
            "max_frames": int(self.config.get("max_frames", 29)),
            "prompt": self.config.get("segmentation_prompt", "robot arm"),
            "confidence": float(self.config.get("confidence", 0.1)),
            "action_metric_weights": self.config.get("action_metric_weights"),
            "af_fdce_ate_norm_scale": float(self.config.get("af_fdce_ate_norm_scale", 0.2)),
            "fdce_scale": float(self.config.get("fdce_scale", 10.0)),
            "fdce_k": int(self.config.get("fdce_k", 16)),
            "fdce_visibility_threshold": float(
                self.config.get("fdce_visibility_threshold", 0.5)
            ),
            "fdce_min_visible_fraction": float(
                self.config.get("fdce_min_visible_fraction", 0.8)
            ),
            "fdce_min_common_frames": int(self.config.get("fdce_min_common_frames", 1)),
            "fdce_seed": int(self.config.get("fdce_seed", 0)),
            "reward_mode": self.config.get("mode", "action"),
            "geometry_enabled": bool(self.config.get("geometry_enabled", False)),
            "all_camera_videos": all_camera_videos,
            "geometry_cameras": cameras,
            "geometry_future_start": int(self.config.get("geometry_future_start", 4)),
            "geometry_future_end": int(self.config.get("geometry_future_end", 28)),
            "geometry_mean_weight": float(self.config.get("geometry_mean_weight", 0.6)),
            "geometry_worst_weight": float(self.config.get("geometry_worst_weight", 0.4)),
            "geometry_psnr_center_db": float(
                self.config.get("geometry_psnr_center_db", 20.4)
            ),
            "geometry_psnr_temperature_db": float(
                self.config.get("geometry_psnr_temperature_db", 1.8)
            ),
            "action_weight": float(self.config.get("action_weight", 1.0)),
            "geometry_weight": float(self.config.get("geometry_weight", 1.0)),
        }
        return gt_path, str(prediction_dir / "head_color.mp4"), kwargs

    @torch.no_grad()
    def score_paths(
        self,
        *,
        condition_id: str,
        prep_dir: str,
        prediction_dir: str | Path,
    ) -> dict[str, Any]:
        gt_path, pred_path, kwargs = self.legacy_kwargs(
            condition_id=condition_id,
            prep_dir=prep_dir,
            prediction_dir=prediction_dir,
        )
        reward = compute_head_reward(gt_path, pred_path, **kwargs)
        if bool(self.config.get("sobel_enabled", False)):
            reward["sobel"] = compute_multiview_sobel_reward(
                kwargs["all_camera_videos"],
                cameras=kwargs["geometry_cameras"],
                future_start=int(self.config.get("sobel_future_start", kwargs["geometry_future_start"])),
                future_end=int(self.config.get("sobel_future_end", kwargs["geometry_future_end"])),
                mean_weight=float(self.config.get("sobel_mean_weight", kwargs["geometry_mean_weight"])),
                worst_weight=float(self.config.get("sobel_worst_weight", kwargs["geometry_worst_weight"])),
                charbonnier_epsilon=float(self.config.get("sobel_charbonnier_epsilon", 1.0e-3)),
            )
        return reward

    @torch.no_grad()
    def score_rollout(
        self, rollout: BranchRollout | OrdinaryRollout, *, prep_dir: str
    ) -> dict[str, Any]:
        reward = self.score_paths(
            condition_id=rollout.group_key.condition_id,
            prep_dir=prep_dir,
            prediction_dir=rollout.seed_dir,
        )
        rollout.reward = reward
        (rollout.seed_dir / "reward.json").write_text(
            json.dumps(reward, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return reward
