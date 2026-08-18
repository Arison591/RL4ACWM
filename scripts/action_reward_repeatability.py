#!/usr/bin/env python3
"""Measure repeatability of the action evaluator on saved rollout videos."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tempflow_video.config import load_tempflow_config
from experiments.tempflow_video.reward_adapter import VideoRewardAdapter
from experiments.action_following.yolo_detector import reset_eef_detector_state


COMPONENTS = (
    "total_reward",
    "final_command_component",
    "combined_raw_command_error",
    "fdce_reward",
    "mean_iou",
    "command_valid_arms",
    "command_coverage",
)


def _row(result: dict[str, Any]) -> dict[str, float]:
    metrics = result.get("action_metrics", {})
    components = result.get("action_reward_components", {}).get("components", {})
    return {
        "total_reward": float(result["total_reward"]),
        "final_command_component": float(metrics["final_command_component"]),
        "combined_raw_command_error": float(metrics["combined_raw_command_error"]),
        "fdce_reward": float(components["fdce"]),
        "mean_iou": float(components["mean_iou"]),
        "command_valid_arms": float(metrics["command_valid_arms"]),
        "command_coverage": float(metrics["command_coverage"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="effective config or config YAML")
    parser.add_argument("--saved-run", required=True, help="run containing branch rollout.json files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=None)
    args = parser.parse_args()
    if args.repeats < 3:
        raise ValueError("repeatability audit requires at least 3 repeats")

    config = load_tempflow_config(args.config)
    reward = VideoRewardAdapter(config["reward"])
    all_rollout_paths = sorted(Path(args.saved_run).glob("**/rollout.json"))[: args.limit]
    rollout_paths = all_rollout_paths[args.offset :: args.stride]
    if not rollout_paths:
        raise FileNotFoundError("no saved rollout.json files found")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for rollout_path in rollout_paths:
        metadata = json.loads(rollout_path.read_text(encoding="utf-8"))
        condition_id = str(metadata["condition_id"])
        prep_dir = Path(config["dataset"]["prep_root"]) / condition_id
        if not prep_dir.is_dir():
            raise FileNotFoundError(f"missing prep directory: {prep_dir}")
        for repeat in range(args.repeats):
            # SAM3 inference states are local to each track_masks invocation;
            # CoWTracker forward is stateless. YOLO owns the only reusable
            # video tracker state, so reset it explicitly around every pass.
            reset_eef_detector_state()
            try:
                result = reward.score_paths(
                    condition_id=condition_id,
                    prep_dir=str(prep_dir),
                    prediction_dir=rollout_path.parent,
                )
            finally:
                reset_eef_detector_state()
            row: dict[str, Any] = {
                "rollout_path": str(rollout_path.parent),
                "condition_id": condition_id,
                "repeat": repeat,
                **_row(result),
            }
            rows.append(row)

    suffix = "" if args.worker_id is None else f"_worker{args.worker_id}"
    csv_path = output / f"reward_repeatability{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_rollout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rollout[str(row["rollout_path"])].append(row)
    floors: dict[str, dict[str, float | int]] = {}
    for component in COMPONENTS:
        differences = []
        for repeats in by_rollout.values():
            for left, right in itertools.combinations(repeats, 2):
                difference = abs(float(left[component]) - float(right[component]))
                if not math.isfinite(difference):
                    raise FloatingPointError(f"non-finite repeatability difference: {component}")
                differences.append(difference)
        floors[component] = {
            "p95_abs_repeat_difference": float(np.percentile(differences, 95)),
            "max_abs_repeat_difference": float(np.max(differences)),
            "pair_count": len(differences),
        }
    report = {
        "saved_run": str(Path(args.saved_run).resolve()),
        "config": str(Path(args.config).resolve()),
        "repeats": args.repeats,
        "rollouts": len(rollout_paths),
        "noise_floors": floors,
        "recommended_component_noise_floors": {
            "command": floors["combined_raw_command_error"]["p95_abs_repeat_difference"],
            "fdce": floors["fdce_reward"]["p95_abs_repeat_difference"],
            "iou": floors["mean_iou"]["p95_abs_repeat_difference"],
        },
    }
    (output / f"reward_noise_floor{suffix}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
