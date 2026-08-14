from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from experiments.action_following.metrics_action_following import compute_all
from experiments.action_following import cowtracker_tracking, fdce_tracks, metrics_fdce, sam_tracking
from experiments.action_following.action_command import action_following_metrics, commanded_trajectory
from experiments.eval_action_following import _load_video, DEFAULT_ARMS
from .reward_functions import action_metrics_to_reward, combine_rewards, compute_geometry_reward


def _resize_video(video: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.stack([cv2.resize(frame, (width, height)) for frame in video])


def prepare_video_pair(gt_path: str, pred_path: str, max_frames: int = 49) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    gt, fps_gt = _load_video(gt_path)
    pred, fps_pred = _load_video(pred_path)
    if max_frames > 0:
        gt, pred = gt[:max_frames], pred[:max_frames]
    if gt.shape[1:3] != pred.shape[1:3]:
        gt = _resize_video(gt, pred.shape[1], pred.shape[2])
    n = min(len(gt), len(pred))
    return gt[:n], pred[:n], {
        "fps_gt": fps_gt,
        "fps_pred": fps_pred,
        "max_frames": max_frames,
        "height": int(pred.shape[1]),
        "width": int(pred.shape[2]),
        "num_frames": int(n),
    }


def _finite_mean(values: list[Any]) -> float | None:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def _fdce_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    prompt: str,
    confidence: float,
    *,
    actions: np.ndarray,
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    k: int = 16,
    fdce_seed: int = 0,
    visibility_threshold: float = 0.5,
    min_visible_fraction: float = 0.8,
    min_common_frames: int = 1,
    masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run the same FDCE metric path as eval_action_following_fdce.py."""
    if not cowtracker_tracking.cowtracker_ready():
        return {"fdce": None, "fdce_error": "CoWTracker is not ready"}
    if masks is None:
        masks = (sam_tracking.track_gt(gt, prompt, confidence), sam_tracking.track_pred(pred, prompt, confidence))
    dense_gt = cowtracker_tracking.track_dense(gt)
    dense_pred = cowtracker_tracking.track_dense(pred)
    results = []
    af_results = []
    height, width = pred.shape[1:3]
    diag = float(np.sqrt(height * height + width * width))
    actions = actions[:len(pred)]
    c2w = c2w[:len(pred)]
    command = {
        arm: commanded_trajectory(
            actions, c2w, intrinsic, sample_size=(height, width), arm=arm
        )
        for arm in DEFAULT_ARMS
    }
    for arm in DEFAULT_ARMS:
        bundle, meta = fdce_tracks.build_track_bundle(
            masks[0], masks[1], dense_gt, dense_pred, k=k, seed=fdce_seed, max_frames=len(gt)
        )
        if meta["init_failure"]:
            continue
        try:
            result = metrics_fdce.foreground_displacement_chamfer_error(
                bundle["generated_tracks"], bundle["reference_tracks"],
                bundle["generated_visibility"], bundle["reference_visibility"],
                visibility_threshold=visibility_threshold,
                min_visible_fraction=min_visible_fraction,
                min_common_frames=min_common_frames,
                return_details=True,
            )
        except (ValueError, FloatingPointError):
            # 单条 rollout 跟踪退化（如生成视频里机械臂不可见，轨迹被可见性过滤掉）
            # 不应让整个 reward / 训练崩溃；跳过该 arm（与 init_failure 同语义）。
            continue
        results.append(result.score)
        try:
            pred_tracks, pred_visibility = fdce_tracks.sample_tracks(
                dense_pred, masks[1], k=k, seed=fdce_seed,
            )
            with np.errstate(all="ignore"):
                centroid = np.nanmean(
                    np.where(pred_visibility[:, :, None], pred_tracks, np.nan), axis=1
                )
        except (ValueError, FloatingPointError):
            continue
        af_results.append(action_following_metrics(centroid, command[arm], diag=diag))
    return {
        "fdce": _finite_mean(results),
        "fdce_valid_arms": len(results),
        "af_fdce_ate": _finite_mean([item.get("af_ate") for item in af_results]),
        "af_fdce_ate_norm": _finite_mean([item.get("af_ate_norm") for item in af_results]),
        "af_fdce_det_coverage": _finite_mean([item.get("af_det_coverage") for item in af_results]),
        "af_fdce_joint_frames": _finite_mean([item.get("af_joint_frames") for item in af_results]),
    }


def compute_head_reward(
    gt_path: str,
    pred_path: str,
    *,
    max_frames: int = 49,
    prompt: str = "robot arm",
    confidence: float = 0.1,
    action_metric_weights: dict[str, float] | None = None,
    af_fdce_ate_norm_scale: float = 0.2,
    fdce_scale: float = 10.0,
    fdce_k: int = 16,
    fdce_visibility_threshold: float = 0.5,
    fdce_min_visible_fraction: float = 0.8,
    fdce_min_common_frames: int = 1,
    fdce_seed: int = 0,
    prep_dir: str | None = None,
    all_camera_videos: dict[str, Any] | None = None,
    geometry_cameras: list[str] | None = None,
    reward_mode: str = "action",
    geometry_enabled: bool = False,
    geometry_future_start: int = 4,
    geometry_future_end: int = 28,
    geometry_mean_weight: float = 0.6,
    geometry_worst_weight: float = 0.4,
    geometry_psnr_center_db: float = 20.4,
    geometry_psnr_temperature_db: float = 1.8,
    action_weight: float = 1.0,
    geometry_weight: float = 1.0,
) -> dict[str, Any]:
    """Compute Action, multi-view PSNR geometry, or their weighted joint reward."""
    reward_mode = str(reward_mode).lower()
    if reward_mode == "psnr":
        reward_mode = "geometry"
    if reward_mode not in {"action", "geometry", "joint"}:
        raise ValueError(f"unknown reward mode: {reward_mode}; expected action/geometry/joint")

    action_metrics = {}
    action_reward = None
    action_components = {"components": {}, "weights": action_metric_weights or {}, "valid": False,
                         "reason": "action reward is disabled by reward mode"}
    protocol = None
    if reward_mode in {"action", "joint"}:
        gt, pred, protocol = prepare_video_pair(gt_path, pred_path, max_frames=max_frames)
        height, width = pred.shape[1:3]
        masks = (
            sam_tracking.track_gt(gt, prompt, confidence),
            sam_tracking.track_pred(pred, prompt, confidence),
        )
        sample_metrics = []
        for arm in DEFAULT_ARMS:
            sample_metrics.append(compute_all(
                gt, pred, prompt, arm, height, width, confidence=confidence, masks=masks
            ))
        action_metrics = {
            key: _finite_mean([item.get(key) for item in sample_metrics])
            for key in ("mean_iou", "ate", "ate_norm", "det_coverage")
        }
        if prep_dir is None:
            raise ValueError("prep_dir is required to compute af_fdce_ate against the action command trajectory")
        actions = np.load(os.path.join(prep_dir, "actions.npy"))
        c2w = np.load(os.path.join(prep_dir, "extrinsic_head.npy"))
        intrinsic = np.load(os.path.join(prep_dir, "intrinsic_head.npy"))
        action_metrics.update(_fdce_metrics(
            gt, pred, prompt, confidence,
            actions=actions, c2w=c2w, intrinsic=intrinsic,
            k=fdce_k,
            fdce_seed=fdce_seed,
            visibility_threshold=fdce_visibility_threshold,
            min_visible_fraction=fdce_min_visible_fraction,
            min_common_frames=fdce_min_common_frames,
            masks=masks,
        ))
        action_reward, action_components = action_metrics_to_reward(
            action_metrics,
            metric_weights=action_metric_weights,
            af_fdce_ate_norm_scale=af_fdce_ate_norm_scale,
            fdce_scale=fdce_scale,
        )

    geometry = compute_geometry_reward(
        all_camera_videos=all_camera_videos,
        camera_names=geometry_cameras or [],
        enabled=bool(geometry_enabled and reward_mode in {"geometry", "joint"}),
        future_start=geometry_future_start,
        future_end=geometry_future_end,
        mean_weight=geometry_mean_weight,
        worst_weight=geometry_worst_weight,
        psnr_center_db=geometry_psnr_center_db,
        psnr_temperature_db=geometry_psnr_temperature_db,
    )
    total_reward = combine_rewards(
        action_reward,
        geometry["reward"],
        mode=reward_mode,
        action_weight=action_weight,
        geometry_weight=geometry_weight,
    )
    if total_reward is not None:
        error = None
    elif reward_mode == "action":
        error = action_components.get("reason")
    elif reward_mode == "geometry":
        error = geometry.get("reason")
    else:
        error = action_components.get("reason") or geometry.get("reason") or "joint reward is unavailable"
    return {
        "total_reward": total_reward,
        "reward_mode": reward_mode,
        "action_reward": action_reward,
        "geometry_reward": geometry["reward"],
        "geometry_enabled": geometry["enabled"],
        "action_metrics": action_metrics,
        "action_reward_components": action_components,
        "geometry": geometry,
        "action_camera": "head",
        "geometry_cameras": geometry_cameras or [],
        "preprocess": protocol,
        "joint_weights": {"action": float(action_weight), "geometry": float(geometry_weight)},
        "valid": total_reward is not None,
        "error": error,
    }
