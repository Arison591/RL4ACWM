#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from experiments.awm_coca.condition_dataset import PrepConditionDataset, build_manifest
from experiments.awm_coca.gesim_runtime import DEFAULT_PROMPT, PersistentGeSimRuntime
from experiments.tempflow_video.config import load_tempflow_config


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _numeric_leaves(value: Any, prefix: str = "reward") -> dict[str, float]:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            output.update(_numeric_leaves(item, f"{prefix}.{key}"))
        return output
    if isinstance(value, list):
        output = {}
        for index, item in enumerate(value):
            output.update(_numeric_leaves(item, f"{prefix}[{index}]"))
        return output
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return {prefix: float(value)}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 2: fixed-seed base inference/reward determinism")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--reward-tolerance", type=float, default=1.0e-8)
    args = parser.parse_args()
    config = load_tempflow_config(args.config)
    os.environ["AWM_ASSET_ROOT"] = str(Path(config["model"]["checkpoint_root"]).parent)
    from experiments.tempflow_video.reward_adapter import VideoRewardAdapter

    manifest, invalid = build_manifest(config["dataset"]["prep_root"], validation_mode="strict")
    if invalid:
        raise RuntimeError(invalid)
    dataset = PrepConditionDataset(manifest)
    raw = dataset[args.condition_index]
    runtime = PersistentGeSimRuntime(config, device="cuda")
    root = Path(args.output)
    artifacts = []
    rewards = []
    adapter = VideoRewardAdapter(config["reward"])
    for name in ("repeat_a", "repeat_b"):
        # Match the production group protocol: every group starts from a fresh
        # PreparedGeSimCondition. Reusing a populated memory-latent cache changes
        # how many random draws the pipeline consumes before future noise.
        prepared = runtime.prepare_condition(raw)
        _, values = runtime.rollout_group(
            prepared,
            seeds=[args.seed],
            output_dir=root / name,
            prompt=DEFAULT_PROMPT,
            expected_group_size=1,
            rollout_batch_size=1,
        )
        artifact = values[0]
        artifacts.append(artifact)
    # Keep the inference repeat isolated from reward-model initialization and
    # CUDA backend changes. Both complete videos are generated first; terminal
    # rewards are then evaluated in the same fixed order.
    for artifact in artifacts:
        rewards.append(
            adapter.score_paths(
                condition_id=raw.condition_id,
                prep_dir=raw.sample_dir,
                prediction_dir=artifact.seed_dir,
            )
        )
    trajectory_hashes = [
        [_tensor_sha(row["latents"]) for row in artifact.trajectory] for artifact in artifacts
    ]
    video_hashes = {
        camera: [_sha(artifact.seed_dir / f"{camera}_color.mp4") for artifact in artifacts]
        for camera in runtime.args.data["train"]["valid_cam"]
    }
    leaves = [_numeric_leaves(value) for value in rewards]
    common = set(leaves[0]).intersection(leaves[1])
    reward_differences = {key: abs(leaves[0][key] - leaves[1][key]) for key in sorted(common)}
    # Legacy plain ATE/ATE-norm diagnostics come from a stateful YOLO tracker
    # but are not inputs to action_metrics_to_reward. Preserve and report their
    # variation without confusing it with terminal reward nondeterminism.
    unused_diagnostics = {"reward.action_metrics.ate", "reward.action_metrics.ate_norm"}
    driving_differences = {
        key: value for key, value in reward_differences.items() if key not in unused_diagnostics
    }
    diagnostic_differences = {
        key: reward_differences[key]
        for key in sorted(unused_diagnostics.intersection(reward_differences))
    }
    report = {
        "ok": (
            trajectory_hashes[0] == trajectory_hashes[1]
            and all(pair[0] == pair[1] for pair in video_hashes.values())
            and max(driving_differences.values(), default=0.0) <= args.reward_tolerance
        ),
        "condition_id": raw.condition_id,
        "seed": args.seed,
        "trajectory_identical": trajectory_hashes[0] == trajectory_hashes[1],
        "trajectory_hashes": trajectory_hashes,
        "video_hashes": video_hashes,
        "reward_tolerance": args.reward_tolerance,
        "max_reward_driving_difference": max(driving_differences.values(), default=0.0),
        "max_all_reward_output_difference": max(reward_differences.values(), default=0.0),
        "unused_diagnostic_differences": diagnostic_differences,
        "rewards": rewards,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "gate_base_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("ok", "condition_id", "seed", "trajectory_identical", "max_reward_driving_difference", "max_all_reward_output_difference")}))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
