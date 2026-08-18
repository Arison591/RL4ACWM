#!/usr/bin/env python3
"""Run the fixed 16-condition x 8-seed base/final paired Action evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.awm_coca.condition_dataset import PrepConditionDataset
from experiments.awm_coca.gesim_runtime import PersistentGeSimRuntime
from experiments.tempflow_video.checkpointing import load_tempflow_checkpoint
from experiments.tempflow_video.config import load_tempflow_config
from experiments.tempflow_video.run import _make_manifest, _seed_everything, evaluate_policy
from experiments.tempflow_video.reward_adapter import VideoRewardAdapter


def _load_policy_only(model: torch.nn.Module, checkpoint: Path) -> int:
    payload = load_tempflow_checkpoint(checkpoint, map_location="cpu")
    policy = payload["policy"]
    if policy["kind"] == "peft":
        from peft import set_peft_model_state_dict

        result = set_peft_model_state_dict(model, policy["state"])
        if getattr(result, "unexpected_keys", None):
            raise ValueError(f"unexpected checkpoint keys: {result.unexpected_keys}")
    elif policy["kind"] == "full":
        model.load_state_dict(policy["state"], strict=True)
    else:
        raise ValueError(f"unsupported checkpoint format: {policy['kind']}")
    return int(payload["trainer_state"]["policy_version"])


def _read_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (str(row["condition_id"]), int(row["seed"]))
        if key in rows:
            raise ValueError(f"duplicate paired evaluation key: {key}")
        rows[key] = row
    return rows


def _metric(row: dict[str, Any], key: str) -> float:
    if key == "total_reward":
        return float(row["reward"]["total_reward"])
    if key == "raw_command_error":
        return -float(row["reward"]["action_metrics"]["combined_raw_command_error"])
    return float(row["reward"]["action_reward_components"]["components"][key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--conditions", type=int, default=16)
    args = parser.parse_args()
    if len(args.seeds) != 8 or len(set(args.seeds)) != 8:
        raise ValueError("paired evaluation requires exactly 8 distinct fixed seeds")
    if args.conditions != 16:
        raise ValueError("paired evaluation requires exactly 16 conditions")

    config = load_tempflow_config(args.config)
    _seed_everything(int(config.get("seed", 42)))
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=False)
    manifest = _make_manifest(config, target)
    dataset = PrepConditionDataset(manifest)
    if len(dataset) < args.conditions:
        raise ValueError(f"need 16 conditions, manifest has {len(dataset)}")
    runtime = PersistentGeSimRuntime(config, device=torch.device("cuda"))
    reward = VideoRewardAdapter(config["reward"])
    evaluate_policy(runtime, dataset, reward, run_dir=target, seeds=args.seeds,
                    max_conditions=args.conditions, tag="base")
    final_version = _load_policy_only(runtime.transformer, Path(args.checkpoint))
    runtime.set_policy_version(final_version)
    evaluate_policy(runtime, dataset, reward, run_dir=target, seeds=args.seeds,
                    max_conditions=args.conditions, tag="final")

    base = _read_rows(target / "evaluation" / "base" / "rewards.jsonl")
    final = _read_rows(target / "evaluation" / "final" / "rewards.jsonl")
    if set(base) != set(final) or len(base) != 128:
        raise RuntimeError("base/final paired evaluation does not contain the required 128 identical keys")
    report: dict[str, Any] = {"pairs": len(base), "final_policy_version": final_version, "metrics": {}}
    for name in ("total_reward", "raw_command_error", "fdce", "mean_iou"):
        differences = {key: _metric(final[key], name) - _metric(base[key], name) for key in base}
        values = np.asarray(list(differences.values()), dtype=np.float64)
        wins = int((values > 0).sum())
        ties = int((values == 0).sum())
        by_condition: dict[str, list[float]] = defaultdict(list)
        by_seed: dict[str, list[float]] = defaultdict(list)
        by_timestep: dict[str, list[float]] = defaultdict(list)
        for key, value in differences.items():
            condition, seed = key
            by_condition[condition].append(value)
            by_seed[str(seed)].append(value)
            # Fixed policy evaluation samples a complete reverse trajectory;
            # unlike branch training it has no single selected transition.
            by_timestep["full_generation"].append(value)
        report["metrics"][name] = {
            "mean_difference": float(values.mean()),
            "median_difference": float(np.median(values)),
            "std_difference": float(values.std()),
            "wins": wins,
            "ties": ties,
            "losses": int((values < 0).sum()),
            "by_condition_mean": {key: float(np.mean(value)) for key, value in by_condition.items()},
            "by_seed_mean": {key: float(np.mean(value)) for key, value in by_seed.items()},
            "source_timestep_mean": {key: float(np.mean(value)) for key, value in by_timestep.items()},
        }
    (target / "paired_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
