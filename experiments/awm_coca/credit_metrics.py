from __future__ import annotations

import math
from typing import Any

import numpy as np


def rank_correlation(predicted, observed) -> dict[str, float | None]:
    x = np.asarray(predicted, dtype=float)
    y = np.asarray(observed, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return {"spearman": None, "pearson": None}
    x, y = x[valid], y[valid]
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return {
        "spearman": float(np.corrcoef(rx, ry)[0, 1]),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
    }

def summarize_credit(credit: dict[str, Any]) -> dict[str, Any]:
    rows = credit.get("step_rows", [])
    weights = [float(row["step_weight"]) for row in rows]
    return {
        "num_steps": len(rows),
        "step_weight_sum": float(sum(weights)),
        "reward_conservation_error": float(credit.get("reward_conservation_error", math.nan)),
        "top_step": rows[int(np.argmax(weights))]["step"] if rows else None,
        "top_step_weight": max(weights) if weights else None,
        "credit_entropy_abs": float(-sum(abs(w) * math.log(abs(w)) for w in weights if abs(w) > 1e-12)) if weights else None,
    }
