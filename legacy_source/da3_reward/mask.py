"""GT-only dynamic masks for the DA3 reward audit.

Head and wrist cameras need different motion models.  Head uses change from
the last observed frame.  Wrist views first remove a robust global affine
camera-motion estimate between consecutive GT frames, then threshold the
residual and retain a short temporal envelope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor


ViewKind = Literal["head", "wrist"]


@dataclass(frozen=True)
class DA3DynamicMaskConfig:
    head_quantile: float = 0.94
    wrist_quantile: float = 0.94
    absolute_rgb_threshold: float = 0.025
    close_kernel: int = 3
    open_kernel: int = 0
    dilation_radius: int = 2
    wrist_temporal_radius: int = 1
    min_component_area: int = 12
    camera_motion: Literal["none", "affine"] = "affine"
    max_corners: int = 800
    corner_quality: float = 0.01
    corner_min_distance: float = 7.0
    ransac_threshold: float = 2.0
    min_affine_inliers: int = 20

    def __post_init__(self) -> None:
        for name in ("head_quantile", "wrist_quantile"):
            value = getattr(self, name)
            if not 0.5 < value < 1.0:
                raise ValueError(f"{name} must be in (0.5, 1.0)")
        if not 0.0 <= self.absolute_rgb_threshold <= 1.0:
            raise ValueError("absolute_rgb_threshold must be in [0, 1]")
        for name in ("close_kernel", "open_kernel"):
            value = getattr(self, name)
            if value not in {0, 1} and value % 2 == 0:
                raise ValueError(f"{name} must be zero/one or an odd number")
        if self.dilation_radius < 0 or self.wrist_temporal_radius < 0:
            raise ValueError("mask radii must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rgb_uint8(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"expected RGB HWC image, got {value.shape}")
    if value.dtype == np.uint8:
        return np.ascontiguousarray(value)
    value = value.astype(np.float32)
    if value.max(initial=0.0) <= 1.0:
        value = value * 255.0
    return np.clip(value, 0, 255).round().astype(np.uint8)


def _global_affine(previous: np.ndarray, current: np.ndarray, config: DA3DynamicMaskConfig) -> tuple[np.ndarray, int]:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=config.max_corners,
        qualityLevel=config.corner_quality,
        minDistance=config.corner_min_distance,
        blockSize=7,
    )
    if corners is None or len(corners) < config.min_affine_inliers:
        return np.eye(2, 3, dtype=np.float32), 0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        corners,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        return np.eye(2, 3, dtype=np.float32), 0
    valid = status.reshape(-1).astype(bool)
    source = corners.reshape(-1, 2)[valid]
    target = tracked.reshape(-1, 2)[valid]
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target = source[finite], target[finite]
    if len(source) < config.min_affine_inliers:
        return np.eye(2, 3, dtype=np.float32), 0
    matrix, inliers = cv2.estimateAffine2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_threshold,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    inlier_count = 0 if inliers is None else int(inliers.sum())
    if matrix is None or inlier_count < config.min_affine_inliers or not np.isfinite(matrix).all():
        return np.eye(2, 3, dtype=np.float32), inlier_count
    return matrix.astype(np.float32), inlier_count


def _residual(
    previous: np.ndarray,
    current: np.ndarray,
    view_kind: ViewKind,
    config: DA3DynamicMaskConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    previous = _rgb_uint8(previous)
    current = _rgb_uint8(current)
    height, width = current.shape[:2]
    if previous.shape != current.shape:
        previous = cv2.resize(previous, (width, height), interpolation=cv2.INTER_AREA)

    matrix = np.eye(2, 3, dtype=np.float32)
    inliers = 0
    if view_kind == "wrist" and config.camera_motion == "affine":
        matrix, inliers = _global_affine(previous, current, config)
    aligned = cv2.warpAffine(
        previous,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.warpAffine(
        np.ones((height, width), dtype=np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    difference = np.mean(np.abs(current.astype(np.float32) - aligned.astype(np.float32)), axis=2) / 255.0
    difference[~valid] = 0.0
    return difference, valid, {"affine": matrix.tolist(), "affine_inliers": inliers}


def _clean_mask(mask: np.ndarray, config: DA3DynamicMaskConfig) -> np.ndarray:
    value = mask.astype(np.uint8)
    if config.close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel))
        value = cv2.morphologyEx(value, cv2.MORPH_CLOSE, kernel)
    if config.open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.open_kernel, config.open_kernel))
        value = cv2.morphologyEx(value, cv2.MORPH_OPEN, kernel)
    if config.min_component_area > 1:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(value, connectivity=8)
        keep = np.zeros_like(value)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= config.min_component_area:
                keep[labels == label] = 1
        value = keep
    if config.dilation_radius:
        size = config.dilation_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        value = cv2.dilate(value, kernel)
    return value.astype(bool)


def _threshold_residual(
    residual: np.ndarray,
    valid: np.ndarray,
    quantile: float,
    config: DA3DynamicMaskConfig,
) -> tuple[np.ndarray, float]:
    samples = residual[valid]
    adaptive = float(np.quantile(samples, quantile)) if samples.size else 1.0
    threshold = max(config.absolute_rgb_threshold, adaptive)
    return _clean_mask((residual >= threshold) & valid, config), threshold


def _temporal_envelope(masks: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return masks
    result = np.zeros_like(masks)
    for frame in range(len(masks)):
        result[frame] = masks[max(0, frame - radius) : min(len(masks), frame + radius + 1)].any(axis=0)
    return result


def make_da3_dynamic_mask(
    gt_rgb: Sequence[Sequence[np.ndarray]],
    observed_last_index: int = 3,
    future_start: int = 4,
    config: DA3DynamicMaskConfig = DA3DynamicMaskConfig(),
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """Build a [V,T,H,W] GT-only mask in head/left-wrist/right-wrist order."""

    if len(gt_rgb) != 3:
        raise ValueError("gt_rgb must contain head, left wrist, and right wrist views")
    if any(len(view) <= future_start for view in gt_rgb):
        raise ValueError("GT clips do not contain the requested future frames")
    frame_count = min(len(view) for view in gt_rgb) - future_start
    all_masks: list[np.ndarray] = []
    view_diagnostics: list[dict[str, Any]] = []
    for view_index, frames in enumerate(gt_rgb):
        kind: ViewKind = "head" if view_index == 0 else "wrist"
        masks, frame_diagnostics = [], []
        for offset in range(frame_count):
            current_index = future_start + offset
            previous_index = observed_last_index if kind == "head" else current_index - 1
            residual, valid, diagnostics = _residual(frames[previous_index], frames[current_index], kind, config)
            quantile = config.head_quantile if kind == "head" else config.wrist_quantile
            mask, threshold = _threshold_residual(residual, valid, quantile, config)
            masks.append(mask)
            frame_diagnostics.append({
                "frame": current_index,
                "reference_frame": previous_index,
                "threshold": threshold,
                "raw_residual_mean": float(residual[valid].mean()) if valid.any() else 0.0,
                "mask_ratio": float(mask.mean()),
                **diagnostics,
            })
        stacked = np.stack(masks)
        if kind == "wrist":
            stacked = _temporal_envelope(stacked, config.wrist_temporal_radius)
        all_masks.append(stacked)
        view_diagnostics.append({
            "view": ("head", "hand_left", "hand_right")[view_index],
            "kind": kind,
            "mean_mask_ratio": float(stacked.mean()),
            "frames": frame_diagnostics,
        })
    result = torch.from_numpy(np.stack(all_masks)).bool()
    if return_diagnostics:
        return result, {"config": config.as_dict(), "views": view_diagnostics}
    return result


__all__ = ["DA3DynamicMaskConfig", "make_da3_dynamic_mask"]
