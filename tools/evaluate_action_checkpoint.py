"""Evaluate one TempFlow policy checkpoint with fixed action-reward seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def _load_policy_checkpoint(runtime, checkpoint: Path) -> int:
    from experiments.tempflow_video.checkpointing import load_tempflow_checkpoint

    payload = load_tempflow_checkpoint(checkpoint, map_location="cpu")
    policy_payload = payload["policy"]
    if policy_payload["kind"] == "peft":
        from peft import set_peft_model_state_dict

        result = set_peft_model_state_dict(runtime.transformer, policy_payload["state"])
        if getattr(result, "unexpected_keys", None):
            raise ValueError(f"unexpected PEFT checkpoint keys: {result.unexpected_keys}")
    elif policy_payload["kind"] == "full":
        runtime.transformer.load_state_dict(policy_payload["state"], strict=True)
    else:
        raise ValueError(f"unknown policy checkpoint kind: {policy_payload['kind']}")
    return int(payload["trainer_state"]["policy_version"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    from experiments.tempflow_video.config import load_tempflow_config

    config = load_tempflow_config(args.config)
    checkpoint_root = Path(config["model"]["checkpoint_root"])
    os.environ.setdefault("AWM_ASSET_ROOT", str(checkpoint_root.parent))
    os.environ["AWM_MODEL_ROOT"] = str(checkpoint_root)
    if not torch.cuda.is_available():
        raise RuntimeError("action checkpoint evaluation requires CUDA")
    torch.cuda.set_device(0)

    # Construct SAM3 before GE-Sim, matching the stable action-training order.
    from experiments.action_following.sam_tracking import get_sam3_video_model

    get_sam3_video_model()

    from experiments.awm_coca.condition_dataset import PrepConditionDataset
    from experiments.awm_coca.gesim_runtime import PersistentGeSimRuntime
    from experiments.tempflow_video.reward_adapter import VideoRewardAdapter
    from experiments.tempflow_video.run import _make_manifest, evaluate_policy

    run_dir = Path(args.output_dir).resolve() / args.label
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest = _make_manifest(config, run_dir)
    dataset = PrepConditionDataset(manifest)
    runtime = PersistentGeSimRuntime(config, device="cuda")
    policy_version = 0
    if args.checkpoint:
        policy_version = _load_policy_checkpoint(runtime, Path(args.checkpoint).resolve())
    runtime.set_policy_version(policy_version)
    reward = VideoRewardAdapter(config["reward"])
    summary = evaluate_policy(
        runtime,
        dataset,
        reward,
        run_dir=run_dir,
        seeds=[int(seed) for seed in config["evaluation"]["fixed_generation_seeds"]],
        max_conditions=int(config["evaluation"].get("fixed_samples", 1)),
        tag=args.label,
    )
    summary.update(
        {
            "label": args.label,
            "checkpoint": None if args.checkpoint is None else str(Path(args.checkpoint).resolve()),
        }
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
