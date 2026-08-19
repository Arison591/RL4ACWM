from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Mapping

import numpy as np


def _frame_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    if gt.shape != pred.shape:
        raise ValueError(f"frame shape mismatch: {gt.shape} != {pred.shape}")
    mse = float(np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2))
    return 10.0 * math.log10((255.0**2) / max(mse, 1e-12))


def aggregate_views(per_view: Mapping[str, float], mean_weight: float = 0.6,
                    worst_weight: float = 0.4) -> tuple[float, float, float]:
    if not per_view:
        raise ValueError("at least one view is required")
    total = float(mean_weight) + float(worst_weight)
    if mean_weight < 0 or worst_weight < 0 or total <= 0:
        raise ValueError("view weights must be nonnegative and nonzero")
    mean, worst = float(np.mean(list(per_view.values()))), float(min(per_view.values()))
    return (mean_weight * mean + worst_weight * worst) / total, mean, worst


def sigmoid_psnr(value: float, center_db: float = 20.4, temperature_db: float = 1.8) -> float:
    logit = (float(value) - center_db) / temperature_db
    return 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1.0 + math.exp(logit))


@dataclass(frozen=True)
class PSNRRewardOutput:
    psnr_per_frame_per_view: dict[str, list[float]]
    psnr_per_view_full: dict[str, float]
    psnr_per_view_future: dict[str, float]
    psnr_aggregate_full_db: float
    psnr_aggregate_future_db: float
    legacy_psnr_sigmoid: float
    mean_view_full_db: float
    worst_view_full_db: float
    mean_view_future_db: float
    worst_view_future_db: float
    history_frames: int

    def as_dict(self) -> dict:
        return asdict(self)


def compute_psnr_reward(gt_by_view: Mapping[str, np.ndarray], pred_by_view: Mapping[str, np.ndarray], *,
                        history_frames: int, mean_weight: float = 0.6, worst_weight: float = 0.4,
                        legacy_future_start: int | None = None, legacy_future_end: int | None = None,
                        center_db: float = 20.4, temperature_db: float = 1.8) -> PSNRRewardOutput:
    if set(gt_by_view) != set(pred_by_view) or not gt_by_view:
        raise ValueError("GT/pred views must be identical and nonempty")
    per_frame = {}
    for view in gt_by_view:
        gt, pred = np.asarray(gt_by_view[view]), np.asarray(pred_by_view[view])
        if gt.shape != pred.shape or gt.ndim < 2:
            raise ValueError(f"invalid clip for {view}")
        per_frame[view] = [_frame_psnr(a, b) for a, b in zip(gt, pred, strict=True)]
    lengths = {len(v) for v in per_frame.values()}
    if len(lengths) != 1:
        raise ValueError("all views must have the same frame count")
    count = next(iter(lengths))
    if not 0 <= int(history_frames) < count:
        raise ValueError("runtime history_frames must select at least one future frame")
    full = {k: float(np.mean(v)) for k, v in per_frame.items()}
    future = {k: float(np.mean(v[history_frames:])) for k, v in per_frame.items()}
    full_agg, full_mean, full_worst = aggregate_views(full, mean_weight, worst_weight)
    future_agg, future_mean, future_worst = aggregate_views(future, mean_weight, worst_weight)
    start = history_frames if legacy_future_start is None else int(legacy_future_start)
    end = count - 1 if legacy_future_end is None else int(legacy_future_end)
    if not 0 <= start <= end < count:
        raise ValueError("invalid legacy frame range")
    legacy_views = {k: float(np.mean(v[start:end + 1])) for k, v in per_frame.items()}
    legacy_agg, _, _ = aggregate_views(legacy_views, mean_weight, worst_weight)
    return PSNRRewardOutput(per_frame, full, future, full_agg, future_agg,
                            sigmoid_psnr(legacy_agg, center_db, temperature_db),
                            full_mean, full_worst, future_mean, future_worst, int(history_frames))

