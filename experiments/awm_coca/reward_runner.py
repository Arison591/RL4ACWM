from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from experiments.action_following.metrics_action_following import compute_all
from experiments.action_following import (
    cowtracker_tracking,
    fdce_tracks,
    metrics_fdce,
    sam_tracking,
    yolo_detector,
)
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


def _arm_specific_action_metrics(
    pred: np.ndarray,
    command: dict[str, np.ndarray],
    *,
    diag: float,
) -> list[dict[str, Any]]:
    """Compare each detector arm track with its matching command trajectory."""
    metrics = []
    for arm in DEFAULT_ARMS:
        try:
            pred_traj, _ = yolo_detector.track_pred(pred, arm)
            metrics.append({
                "arm": arm,
                **action_following_metrics(pred_traj, command[arm], diag=diag),
            })
        except (ValueError, FloatingPointError):
            continue
    return metrics


def _aggregate_arm_command_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose the exact uncompressed two-arm command error used for ranking."""
    by_arm = {str(item.get("arm")): item for item in metrics}
    frame_errors: dict[int, list[float]] = {}
    for item in metrics:
        for frame, error in zip(
            item.get("af_frame_indices", ()), item.get("af_per_frame_error", ())
        ):
            if np.isfinite(error):
                frame_errors.setdefault(int(frame), []).append(float(error))
    per_frame = {
        str(frame): float(np.mean(errors)) for frame, errors in sorted(frame_errors.items())
    }
    all_errors = [error for errors in frame_errors.values() for error in errors]
    left = by_arm.get("left", {})
    right = by_arm.get("right", {})
    valid_arms = sum(
        int(np.isfinite(float(item.get("af_ate", np.nan)))) for item in (left, right)
    )
    coverage = _finite_mean([
        left.get("af_det_coverage"), right.get("af_det_coverage")
    ])
    arm_errors = [
        float(item["af_ate"])
        for item in (left, right)
        if item.get("af_ate") is not None and np.isfinite(item["af_ate"])
    ]
    return {
        "left_arm_raw_error": left.get("af_ate"),
        "right_arm_raw_error": right.get("af_ate"),
        "per_frame_raw_error": per_frame,
        "combined_raw_command_error": (
            float(np.mean(all_errors)) if all_errors else _finite_mean(arm_errors)
        ),
        "command_valid_arms": valid_arms,
        "command_coverage": coverage,
        "command_is_complete": bool(valid_arms == len(DEFAULT_ARMS) and coverage == 1.0),
    }


def _track_sam_pair(
    gt: np.ndarray,
    pred: np.ndarray,
    prompt: str,
    confidence: float,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    gt_masks, gt_diagnostics = sam_tracking.track_gt(
        gt,
        prompt,
        confidence,
        return_diagnostics=True,
    )
    pred_masks, pred_diagnostics = sam_tracking.track_pred(
        pred,
        prompt,
        confidence,
        return_diagnostics=True,
    )
    return (gt_masks, pred_masks), {
        "reference": asdict(gt_diagnostics),
        "generated": asdict(pred_diagnostics),
    }


def _fdce_only_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    masks: tuple[np.ndarray, np.ndarray],
    k: int,
    fdce_seed: int,
    visibility_threshold: float,
    min_visible_fraction: float,
    min_common_frames: int,
) -> dict[str, Any]:
    """Compute FDCE without invoking unrelated IoU, command, or YOLO paths."""
    if not cowtracker_tracking.cowtracker_ready():
        return {"fdce": None, "fdce_error": "CoWTracker is not ready", "fdce_valid": False}

    dense_gt = cowtracker_tracking.track_dense(gt)
    dense_pred = cowtracker_tracking.track_dense(pred)
    try:
        bundle, meta = fdce_tracks.build_track_bundle(
            masks[0],
            masks[1],
            dense_gt,
            dense_pred,
            k=k,
            seed=fdce_seed,
            max_frames=len(gt),
        )
        if meta["init_failure"]:
            return {
                "fdce": None,
                "fdce_error": "no foreground anchors survive on frame 0",
                "fdce_valid": False,
                "fdce_track_meta": meta,
            }
        result = metrics_fdce.foreground_displacement_chamfer_error(
            bundle["generated_tracks"],
            bundle["reference_tracks"],
            bundle["generated_visibility"],
            bundle["reference_visibility"],
            visibility_threshold=visibility_threshold,
            min_visible_fraction=min_visible_fraction,
            min_common_frames=min_common_frames,
            return_details=True,
        )
    except (ValueError, FloatingPointError) as exc:
        return {
            "fdce": None,
            "fdce_error": f"{type(exc).__name__}: {exc}",
            "fdce_valid": False,
        }

    return {
        "fdce": float(result.score),
        "fdce_valid": True,
        # Preserve the legacy key while correcting its meaning: FDCE is one
        # global foreground score, not one score per robot arm.
        "fdce_valid_arms": 1,
        "fdce_generated_to_reference": float(result.generated_to_reference),
        "fdce_reference_to_generated": float(result.reference_to_generated),
        "fdce_generated_tracks": int(result.generated_tracks),
        "fdce_reference_tracks": int(result.reference_tracks),
        "fdce_valid_pairs": int(result.valid_pairs),
        "fdce_track_meta": meta,
    }


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
    if masks is None:
        masks, _ = _track_sam_pair(gt, pred, prompt, confidence)
    fdce_metrics = _fdce_only_metrics(
        gt,
        pred,
        masks=masks,
        k=k,
        fdce_seed=fdce_seed,
        visibility_threshold=visibility_threshold,
        min_visible_fraction=min_visible_fraction,
        min_common_frames=min_common_frames,
    )
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
    # Action-following must be arm-specific.  A centroid of the merged
    # two-arm mask cannot match both command trajectories: it is bounded below
    # by roughly half the left/right command separation.  YOLO already exposes
    # stable left/right EEF tracks, so compare each class with its own command.
    af_results = _arm_specific_action_metrics(pred, command, diag=diag)
    command_metrics = _aggregate_arm_command_metrics(af_results)
    return {
        **fdce_metrics,
        "af_fdce_valid_arms": len(af_results),
        "af_fdce_ate": command_metrics["combined_raw_command_error"],
        "af_fdce_ate_norm": _finite_mean([item.get("af_ate_norm") for item in af_results]),
        "af_fdce_det_coverage": _finite_mean([item.get("af_det_coverage") for item in af_results]),
        "af_fdce_joint_frames": _finite_mean([item.get("af_joint_frames") for item in af_results]),
        **command_metrics,
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
        masks, sam_diagnostics = _track_sam_pair(gt, pred, prompt, confidence)
        positive_metrics = {
            key
            for key, value in (action_metric_weights or {}).items()
            if float(value) > 0.0
        }
        fdce_only = positive_metrics == {"fdce"}
        if fdce_only:
            action_metrics = _fdce_only_metrics(
                gt,
                pred,
                masks=masks,
                k=fdce_k,
                fdce_seed=fdce_seed,
                visibility_threshold=fdce_visibility_threshold,
                min_visible_fraction=fdce_min_visible_fraction,
                min_common_frames=fdce_min_common_frames,
            )
        else:
            height, width = pred.shape[1:3]
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
        action_metrics["sam_mask_diagnostics"] = sam_diagnostics
        action_reward, action_components = action_metrics_to_reward(
            action_metrics,
            metric_weights=action_metric_weights,
            af_fdce_ate_norm_scale=af_fdce_ate_norm_scale,
            fdce_scale=fdce_scale,
        )
        if not fdce_only:
            action_metrics["final_command_component"] = action_components["components"].get(
                "af_fdce_ate_norm"
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
