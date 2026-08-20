#!/usr/bin/env python3
"""Audit FDCE-only reward coverage for a fixed condition list and rollouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from experiments.awm_coca.reward_runner import compute_head_reward


def _ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _score(condition_id: str, gt_path: Path, pred_path: Path) -> dict:
    result = compute_head_reward(
        str(gt_path),
        str(pred_path),
        max_frames=29,
        prompt="robot arm",
        confidence=0.1,
        action_metric_weights={"fdce": 1.0},
        fdce_scale=10.0,
        reward_mode="action",
        geometry_enabled=False,
        prep_dir=None,
    )
    metrics = result.get("action_metrics", {})
    sam = metrics.get("sam_mask_diagnostics", {})
    return {
        "condition_id": condition_id,
        "gt_path": str(gt_path),
        "pred_path": str(pred_path),
        "valid": bool(result.get("valid")),
        "fdce": metrics.get("fdce"),
        "fdce_reward": result.get("total_reward"),
        "fdce_error": metrics.get("fdce_error"),
        "generated_initial_mask_recovered": bool(
            sam.get("generated", {}).get("recovered_initial_frame", False)
        ),
        "generated_recovery_prompt_frame": sam.get("generated", {}).get(
            "recovery_prompt_frame"
        ),
        "reference_initial_mask_recovered": bool(
            sam.get("reference", {}).get("recovered_initial_frame", False)
        ),
        "track_meta": metrics.get("fdce_track_meta"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument(
        "--pred-template",
        required=True,
        help="Format string with {condition_id}, e.g. /.../{condition_id}/seed_1/videos/head.mp4",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for index, condition_id in enumerate(_ids(args.ids_file), 1):
        gt_path = args.gt_root / condition_id / "head_29_frames.mp4"
        pred_path = Path(args.pred_template.format(condition_id=condition_id))
        if not gt_path.is_file() or not pred_path.is_file():
            row = {
                "condition_id": condition_id,
                "valid": False,
                "fdce": None,
                "fdce_reward": None,
                "fdce_error": "missing GT or prediction video",
                "gt_path": str(gt_path),
                "pred_path": str(pred_path),
            }
        else:
            try:
                row = _score(condition_id, gt_path, pred_path)
            except Exception as exc:  # keep auditing the remaining fixed set
                row = {
                    "condition_id": condition_id,
                    "valid": False,
                    "fdce": None,
                    "fdce_reward": None,
                    "fdce_error": f"{type(exc).__name__}: {exc}",
                    "gt_path": str(gt_path),
                    "pred_path": str(pred_path),
                }
        rows.append(row)
        print(json.dumps({"index": index, **row}, ensure_ascii=False), flush=True)

    valid = [
        row
        for row in rows
        if row["valid"]
        and isinstance(row["fdce"], (int, float))
        and math.isfinite(float(row["fdce"]))
    ]
    fdce = [float(row["fdce"]) for row in valid]
    rewards = [float(row["fdce_reward"]) for row in valid]
    summary = {
        "conditions": len(rows),
        "valid_conditions": len(valid),
        "invalid_conditions": len(rows) - len(valid),
        "generated_initial_masks_recovered": sum(
            bool(row.get("generated_initial_mask_recovered")) for row in valid
        ),
        "fdce": None
        if not fdce
        else {"min": min(fdce), "mean": mean(fdce), "max": max(fdce)},
        "fdce_reward": None
        if not rewards
        else {"min": min(rewards), "mean": mean(rewards), "max": max(rewards)},
    }
    payload = {"summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
