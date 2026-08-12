from __future__ import annotations

from typing import Any

import numpy as np


def compare_reward_drop(original_reward: float, intervened_rewards: dict[int, float]) -> list[dict[str, Any]]:
    """Convert paired intervention runs into diagnostic reward drops.

    The module does not invent an intervention. The caller must provide
    intervention results produced by a controlled rerun of the same rollout.
    """
    return [
        {
            "window_id": int(window_id),
            "original_reward": float(original_reward),
            "intervened_reward": float(value),
            "reward_drop": float(original_reward - value),
        }
        for window_id, value in sorted(intervened_rewards.items())
    ]


def evaluate_prediction(predicted_credit: list[float], reward_drop: list[float]) -> dict[str, float | None]:
    x = np.asarray(predicted_credit, dtype=float)
    y = np.asarray(reward_drop, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return {"spearman": None, "pearson": None, "top1_match": None}
    x, y = x[valid], y[valid]
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return {
        "spearman": float(np.corrcoef(rx, ry)[0, 1]),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "top1_match": float(np.argmax(x) == np.argmax(y)),
    }
