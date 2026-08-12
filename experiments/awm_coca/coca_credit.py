from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    value = torch.nn.functional.cosine_similarity(a, b, dim=1).mean()
    return float(value.item())


def _window_weights(similarities: np.ndarray, window_size: int) -> np.ndarray:
    """CoCA window contribution, followed by step-level normalization."""
    # similarities contains z_0 ... z_K; each contribution is assigned to step 1..K.
    increments = similarities[1:] - similarities[:-1]
    k = len(increments)
    if k == 0:
        return np.empty(0, dtype=np.float64)
    window_size = max(int(window_size), 1)
    weights = np.zeros(k, dtype=np.float64)
    window_means = []
    ranges = []
    for start in range(0, k, window_size):
        end = min(start + window_size, k)
        window_means.append(float(similarities[start + 1:end + 1].mean()))
        ranges.append((start, end))
    previous = float(similarities[0])
    contributions = []
    for mean in window_means:
        contributions.append(mean - previous)
        previous = mean
    for (start, end), contribution in zip(ranges, contributions):
        weights[start:end] = contribution
    denominator = float(weights.sum())
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return np.full(k, 1.0 / k, dtype=np.float64)
    return weights / denominator


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-8)
    x = values / temperature
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / exp.sum()


def compute_credit(
    trajectory: dict[str, Any],
    total_reward: float,
    *,
    advantage: float | None = None,
    window_size: int = 3,
    num_training_noise_levels: int = 4,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Compute post-hoc reverse-step and training-bin credit.

    The output is diagnostic only; it does not sample a training noise level or
    perform any model update.
    """
    step_rows: list[dict[str, Any]] = []
    chunk_rows = []
    for chunk_id, chunk in enumerate(trajectory.get("chunks", [])):
        if len(chunk) < 2:
            continue
        final = chunk[-1]["latents"]
        similarities = np.asarray([_cosine(item["latents"], final) for item in chunk])
        weights = _window_weights(similarities, window_size)
        for local_step, weight in enumerate(weights, start=1):
            item = chunk[local_step]
            step_rows.append({
                "chunk": chunk_id,
                "step": int(item["step"]),
                "local_step": local_step,
                "timestep": float(item.get("timestep", 0.0)),
                "cosine_similarity": float(similarities[local_step]),
                "delta_similarity": float(similarities[local_step] - similarities[local_step - 1]),
                "step_weight_raw": float(weight),
            })
        chunk_rows.append((chunk_id, similarities, weights))

    if not step_rows:
        raise ValueError("trajectory contains no complete denoise chunk")
    raw = np.asarray([row["step_weight_raw"] for row in step_rows], dtype=np.float64)
    denom = float(raw.sum())
    step_weights = raw / denom if np.isfinite(denom) and abs(denom) > 1e-12 else np.full_like(raw, 1.0 / len(raw))
    for row, weight in zip(step_rows, step_weights):
        row["step_weight"] = float(weight)
        row["step_reward"] = float(total_reward * weight)
        row["step_advantage"] = None if advantage is None else float(advantage * weight)

    levels = max(int(num_training_noise_levels), 1)
    bins = [[] for _ in range(levels)]
    for index, row in enumerate(step_rows):
        bin_id = min((index * levels) // len(step_rows), levels - 1)
        row["noise_level"] = int(bin_id + 1)
        bins[bin_id].append(index)
    noise_scores = np.asarray([raw[idxs].sum() for idxs in bins], dtype=np.float64)
    # Use the CoCA proposal directly as q(k). AWM's original p(k) is not
    # mixed in during this evaluation-only phase.
    q = _softmax(noise_scores, temperature)
    noise_rows = []
    for level, idxs in enumerate(bins, start=1):
        noise_rows.append({
            "noise_level": level,
            "reverse_step_start": int(min(step_rows[i]["step"] for i in idxs)),
            "reverse_step_end": int(max(step_rows[i]["step"] for i in idxs)),
            "noise_score": float(noise_scores[level - 1]),
            "noise_weight": float(sum(step_weights[i] for i in idxs)),
            "noise_level_reward": float(total_reward * sum(step_weights[i] for i in idxs)),
            "q": float(q[level - 1]),
        })
    return {
        "total_reward": float(total_reward),
        "advantage": None if advantage is None else float(advantage),
        "num_reverse_steps": len(step_rows),
        "num_training_noise_levels": levels,
        "window_size": int(window_size),
        "step_weight_sum": float(sum(row["step_weight"] for row in step_rows)),
        "reward_conservation_error": float(abs(sum(row["step_reward"] for row in step_rows) - total_reward)),
        "step_rows": step_rows,
        "noise_rows": noise_rows,
    }
