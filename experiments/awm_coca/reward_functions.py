from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


DEFAULT_GEOMETRY_CAMERAS = ("head", "hand_left", "hand_right")


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _higher_is_better(value: Any) -> float | None:
    value = _safe_float(value)
    return None if value is None else max(0.0, min(1.0, value))


def _lower_is_better(value: Any, scale: float) -> float | None:
    value = _safe_float(value)
    if value is None:
        return None
    return 1.0 - max(0.0, min(1.0, value / max(float(scale), 1e-8)))


def _read_video(path: str) -> list[np.ndarray]:
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


def _view_psnr(
    gt_path: str,
    pred_path: str,
    *,
    future_start: int,
    future_end: int,
) -> tuple[float, list[float]]:
    gt_frames = _read_video(gt_path)
    pred_frames = _read_video(pred_path)
    required = future_end + 1
    if len(gt_frames) < required or len(pred_frames) < required:
        raise ValueError(
            f"expected at least {required} frames, got gt={len(gt_frames)} pred={len(pred_frames)}"
        )
    frame_scores = []
    for gt, pred in zip(
        gt_frames[future_start : future_end + 1],
        pred_frames[future_start : future_end + 1],
        strict=True,
    ):
        if gt.shape[:2] != pred.shape[:2]:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)
        mse = float(np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2))
        frame_scores.append(10.0 * math.log10((255.0**2) / max(mse, 1.0e-12)))
    return float(np.mean(frame_scores)), frame_scores


def _psnr_to_reward(psnr_db: float, *, center_db: float, temperature_db: float) -> float:
    if not math.isfinite(center_db):
        raise ValueError(f"PSNR center must be finite, got {center_db}")
    if not math.isfinite(temperature_db) or temperature_db <= 0.0:
        raise ValueError(f"PSNR temperature must be finite and positive, got {temperature_db}")
    logit = (float(psnr_db) - float(center_db)) / float(temperature_db)
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def compute_geometry_reward(
    *,
    all_camera_videos: dict[str, Any] | None = None,
    camera_names: list[str] | None = None,
    enabled: bool = False,
    future_start: int = 4,
    future_end: int = 28,
    mean_weight: float = 0.6,
    worst_weight: float = 0.4,
    psnr_center_db: float = 20.4,
    psnr_temperature_db: float = 1.8,
    **kwargs,
):
    """计算多视角 RGB PSNR geometry reward。"""
    cameras = list(camera_names or DEFAULT_GEOMETRY_CAMERAS)
    base = {
        "enabled": bool(enabled),
        "reward": None,
        "metrics": {},
        "camera_names": cameras,
        "received_multi_camera": bool(all_camera_videos),
        "reason": None,
    }
    if not enabled:
        return {**base, "reason": "geometry reward is disabled"}
    if future_start < 0 or future_end < future_start:
        return {**base, "reason": "frame range must satisfy 0 <= future_start <= future_end"}
    if not math.isfinite(mean_weight) or not math.isfinite(worst_weight):
        return {**base, "reason": "PSNR aggregation weights must be finite"}
    if mean_weight < 0.0 or worst_weight < 0.0 or mean_weight + worst_weight <= 0.0:
        return {**base, "reason": "PSNR aggregation weights must be non-negative and not both zero"}
    if not all_camera_videos:
        return {**base, "reason": "multi-camera GT/pred videos are required"}
    missing = [camera for camera in cameras if camera not in all_camera_videos]
    if missing:
        return {**base, "reason": f"missing camera videos: {missing}"}

    try:
        per_view = {}
        per_frame = {}
        for camera in cameras:
            pair = all_camera_videos[camera]
            score, frame_scores = _view_psnr(
                pair["gt"],
                pair["pred"],
                future_start=future_start,
                future_end=future_end,
            )
            per_view[camera] = score
            per_frame[camera] = frame_scores
        weight_sum = mean_weight + worst_weight
        normalized_mean_weight = float(mean_weight / weight_sum)
        normalized_worst_weight = float(worst_weight / weight_sum)
        mean_psnr_db = float(np.mean(list(per_view.values())))
        worst_view_psnr_db = float(min(per_view.values()))
        balanced_psnr_db = (
            normalized_mean_weight * mean_psnr_db
            + normalized_worst_weight * worst_view_psnr_db
        )
        reward = _psnr_to_reward(
            balanced_psnr_db,
            center_db=psnr_center_db,
            temperature_db=psnr_temperature_db,
        )
    except Exception as exc:
        return {**base, "reason": f"PSNR geometry reward failed: {exc}"}

    return {
        **base,
        "reward": float(reward),
        "metrics": {
            "type": "rgb_psnr",
            "per_view_psnr_db": per_view,
            "per_frame_psnr_db": per_frame,
            "mean_psnr_db": mean_psnr_db,
            "worst_view_psnr_db": worst_view_psnr_db,
            "balanced_psnr_db": balanced_psnr_db,
            "mean_weight": normalized_mean_weight,
            "worst_weight": normalized_worst_weight,
            "psnr_center_db": float(psnr_center_db),
            "psnr_temperature_db": float(psnr_temperature_db),
            "normalization": "sigmoid((balanced_psnr_db - psnr_center_db) / psnr_temperature_db)",
            "future_start": int(future_start),
            "future_end": int(future_end),
            "frame_count_per_view": int(future_end - future_start + 1),
        },
    }


# 兼容已有调用方；新代码使用不带 placeholder 的正式名称。
compute_geometry_reward_placeholder = compute_geometry_reward


def action_metrics_to_reward(
    metrics: dict[str, Any],
    *,
    metric_weights: dict[str, float] | None = None,
    af_fdce_ate_norm_scale: float = 0.2,
    fdce_scale: float = 10.0,
) -> tuple[float | None, dict[str, Any]]:
    weights = metric_weights or {"mean_iou": 1 / 3, "af_fdce_ate_norm": 1 / 3, "fdce": 1 / 3}
    values = {
        "mean_iou": _higher_is_better(metrics.get("mean_iou")),
        "af_fdce_ate_norm": _lower_is_better(
            metrics.get("af_fdce_ate_norm"),
            af_fdce_ate_norm_scale,
        ),
        "fdce": _lower_is_better(metrics.get("fdce"), fdce_scale),
    }
    required = {key for key in values if float(weights.get(key, 0.0)) > 0.0}
    missing = sorted(key for key in required if values[key] is None)
    if missing:
        return None, {
            "components": values,
            "weights": weights,
            "valid": False,
            "reason": "required action-following metrics are unavailable",
            "missing_metrics": missing,
        }
    total_weight = sum(float(weights.get(key, 0.0)) for key in values)
    if total_weight <= 0:
        return None, {"components": values, "weights": weights, "valid": False, "reason": "metric weights sum to zero"}
    reward = sum(
        float(weights.get(key, 0.0)) * float(values[key])
        for key in values
        if float(weights.get(key, 0.0)) != 0.0
    ) / total_weight
    return float(reward), {"components": values, "weights": weights, "valid": True}


def combine_rewards(
    action_reward: float | None,
    geometry_reward: float | None = None,
    *,
    mode: str = "action",
    action_weight: float = 1.0,
    geometry_weight: float = 1.0,
) -> float | None:
    mode = str(mode).lower()
    if mode == "action":
        return action_reward
    if mode in {"geometry", "psnr"}:
        return geometry_reward
    if mode != "joint":
        raise ValueError(f"unknown reward mode: {mode}; expected action/geometry/joint")
    if action_reward is None or geometry_reward is None:
        return None
    if not math.isfinite(action_weight) or not math.isfinite(geometry_weight):
        raise ValueError("joint reward weights must be finite")
    if action_weight < 0.0 or geometry_weight < 0.0 or action_weight + geometry_weight <= 0.0:
        raise ValueError("joint reward weights must be non-negative and not both zero")
    return float(
        (action_weight * action_reward + geometry_weight * geometry_reward)
        / (action_weight + geometry_weight)
    )
