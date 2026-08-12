from __future__ import annotations

import json
import os
from typing import Any

import torch


def save_trajectory(payload: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def load_trajectory(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_metadata(metadata: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, allow_nan=True)


def flatten_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all saved steps while retaining chunk/step identity."""
    result = []
    for chunk_id, chunk in enumerate(payload.get("chunks", [])):
        for item in chunk:
            row = dict(item)
            row["chunk"] = chunk_id
            result.append(row)
    return result
