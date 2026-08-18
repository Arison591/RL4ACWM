#!/usr/bin/env python3
"""Merge repeatability worker CSVs into the canonical noise-floor report."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

COMPONENTS = (
    "total_reward", "final_command_component", "combined_raw_command_error",
    "fdce_reward", "mean_iou", "command_valid_arms", "command_coverage",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted(Path(args.input_dir).glob("reward_repeatability_worker*.csv"))
    if not paths:
        raise FileNotFoundError("no repeatability worker CSVs found")
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    by_rollout = defaultdict(list)
    for row in rows:
        by_rollout[row["rollout_path"]].append(row)
    floors = {}
    for component in COMPONENTS:
        differences = []
        for repeats in by_rollout.values():
            for left, right in itertools.combinations(repeats, 2):
                value = abs(float(left[component]) - float(right[component]))
                if not math.isfinite(value):
                    raise FloatingPointError(f"non-finite difference: {component}")
                differences.append(value)
        floors[component] = {
            "p95_abs_repeat_difference": float(np.percentile(differences, 95)),
            "max_abs_repeat_difference": float(np.max(differences)),
            "pair_count": len(differences),
        }
    report = {
        "worker_csvs": [str(path) for path in paths],
        "repeats": len(next(iter(by_rollout.values()))),
        "rollouts": len(by_rollout),
        "condition_count": len({row["condition_id"] for row in rows}),
        "noise_floors": floors,
        "recommended_component_noise_floors": {
            "command": floors["combined_raw_command_error"]["p95_abs_repeat_difference"],
            "fdce": floors["fdce_reward"]["p95_abs_repeat_difference"],
            "iou": floors["mean_iou"]["p95_abs_repeat_difference"],
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
