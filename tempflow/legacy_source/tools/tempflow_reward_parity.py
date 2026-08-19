#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from experiments.tempflow_video.config import load_tempflow_config


def _compare(left: Any, right: Any, path: str, errors: list[str], tolerance: float) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            errors.append(f"{path}: key mismatch {left.keys()} != {right.keys()}")
            return
        for key in left:
            _compare(left[key], right[key], f"{path}.{key}", errors, tolerance)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            errors.append(f"{path}: length mismatch {len(left)} != {len(right)}")
            return
        for index, (a, b) in enumerate(zip(left, right)):
            _compare(a, b, f"{path}[{index}]", errors, tolerance)
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not (math.isfinite(float(left)) and math.isfinite(float(right))):
            if left != right:
                errors.append(f"{path}: non-finite mismatch {left!r} != {right!r}")
        elif abs(float(left) - float(right)) > tolerance:
            errors.append(f"{path}: |{left!r}-{right!r}| > {tolerance}")
        return
    if left != right:
        errors.append(f"{path}: {left!r} != {right!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare real legacy reward and VideoRewardAdapter outputs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--adapter-json",
        help="existing reward.json produced by VideoRewardAdapter (avoids recomputing that path)",
    )
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    args = parser.parse_args()
    config = load_tempflow_config(args.config)
    os.environ["AWM_ASSET_ROOT"] = str(Path(config["model"]["checkpoint_root"]).parent)
    from experiments.awm_coca.reward_runner import compute_head_reward
    from experiments.tempflow_video.reward_adapter import VideoRewardAdapter

    adapter = VideoRewardAdapter(config["reward"])
    prep_dir = str(Path(config["dataset"]["prep_root"]) / args.condition_id)
    adapted = (
        json.loads(Path(args.adapter_json).read_text(encoding="utf-8"))
        if args.adapter_json
        else adapter.score_paths(
            condition_id=args.condition_id,
            prep_dir=prep_dir,
            prediction_dir=args.prediction_dir,
        )
    )
    gt, pred, kwargs = adapter.legacy_kwargs(
        condition_id=args.condition_id,
        prep_dir=prep_dir,
        prediction_dir=args.prediction_dir,
    )
    legacy = compute_head_reward(gt, pred, **kwargs)
    errors: list[str] = []
    _compare(adapted, legacy, "reward", errors, args.tolerance)
    report = {
        "ok": not errors,
        "condition_id": args.condition_id,
        "prediction_dir": str(Path(args.prediction_dir).resolve()),
        "tolerance": args.tolerance,
        "adapter_result_source": str(Path(args.adapter_json).resolve()) if args.adapter_json else "live",
        "errors": errors,
        "adapter": adapted,
        "legacy": legacy,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "errors": len(errors), "output": str(target)}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
