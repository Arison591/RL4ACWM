from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.awm_coca.condition_dataset import PrepConditionDataset, build_manifest, write_manifest
from experiments.awm_coca.gesim_runtime import DEFAULT_PROMPT, PersistentGeSimRuntime
from experiments.tempflow_video.advantage import standardize_group_rewards
from experiments.tempflow_video.checkpointing import (
    load_tempflow_checkpoint,
    restore_tempflow_checkpoint,
    save_tempflow_checkpoint,
)
from experiments.tempflow_video.config import dump_effective_config, load_tempflow_config
from experiments.tempflow_video.distributed import (
    DistributedContext,
    partition_indices,
    weighted_mean_dict,
)
from experiments.tempflow_video.policy import ReferencePolicyAdapter, VideoPolicyAdapter
from experiments.tempflow_video.preflight import run_preflight, write_preflight_report
from experiments.tempflow_video.sampler import FlowGRPOVideoSampler, TempFlowBranchSampler
from experiments.tempflow_video.schemas import BranchRollout
from experiments.tempflow_video.trainer import TempFlowOptimizerConfig, TempFlowVideoTrainer


def _jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _numeric_reward_leaves(value: Any, prefix: str = "reward") -> dict[str, float]:
    if isinstance(value, dict):
        output: dict[str, float] = {}
        for key, item in value.items():
            output.update(_numeric_reward_leaves(item, f"{prefix}.{key}"))
        return output
    if isinstance(value, list):
        output = {}
        for index, item in enumerate(value):
            output.update(_numeric_reward_leaves(item, f"{prefix}[{index}]"))
        return output
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return {prefix: float(value)}
    return {}


def _make_manifest(
    config: dict[str, Any], run_dir: Path, *, write_files: bool = True
) -> dict[str, Any]:
    dataset = config["dataset"]
    manifest, invalid = build_manifest(
        dataset["prep_root"],
        include_samples=dataset.get("include_samples", ()),
        exclude_samples=dataset.get("exclude_samples", ()),
        limit=int(dataset.get("limit", 0)),
        validation_mode=dataset.get("validation_mode", "strict"),
    )
    if write_files:
        write_manifest(manifest, invalid, run_dir)
    return manifest


def _optimizer_config(config: dict[str, Any]) -> TempFlowOptimizerConfig:
    value = config["optimizer"]
    return TempFlowOptimizerConfig(
        learning_rate=float(value["learning_rate"]),
        betas=tuple(value.get("betas", (0.9, 0.999))),
        epsilon=float(value.get("epsilon", 1.0e-8)),
        weight_decay=float(value.get("weight_decay", 0.0)),
        max_grad_norm=float(value.get("max_grad_norm", 1.0)),
        clip_range=float(value.get("clip_range", 1.0e-4)),
        kl_beta=float(value.get("reference_kl_beta", 0.01)),
        warmup_steps=int(value.get("warmup_steps", 0)),
        log_term_grad_norm=bool(value.get("log_term_grad_norm", True)),
    )


def _assert_resume_compatible(saved: dict[str, Any], current: dict[str, Any]) -> None:
    # These fields define rollout distribution, reward semantics and optimizer
    # state. Logging/checkpoint cadence and the requested stopping step may vary.
    saved = json.loads(json.dumps(saved, default=str))
    saved_reward = saved.setdefault("reward", {})
    saved_reward.setdefault("time_alignment_protocol", "legacy_frame_index_truncate")
    saved_reward.setdefault("generated_fps", 16)
    saved_reward.setdefault("expected_gt_fps", 30)
    for key in (
        "dataset",
        "model",
        "rollout",
        "reward",
        "tempflow",
        "optimizer",
        "distributed",
    ):
        if saved.get(key) != current.get(key):
            raise ValueError(f"resume config mismatch in {key}; refusing mixed-policy training")


def _new_run_dir(config: dict[str, Any], resume: str | None) -> Path:
    if resume:
        checkpoint = Path(resume).resolve()
        if checkpoint.parent.name == "checkpoints":
            return checkpoint.parent.parent
        raise ValueError("resume checkpoint must be under <run>/checkpoints/checkpoint_N")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config["output_dir"]) / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def evaluate_policy(
    runtime: PersistentGeSimRuntime,
    dataset: PrepConditionDataset,
    reward: VideoRewardAdapter,
    *,
    run_dir: Path,
    seeds: list[int],
    max_conditions: int,
    tag: str,
) -> dict[str, Any]:
    rows = []
    pending = []
    target = run_dir / "evaluation" / tag
    for index in range(min(max_conditions, len(dataset))):
        raw = dataset[index]
        prepared = runtime.prepare_condition(raw)
        _, artifacts = runtime.rollout_group(
            prepared,
            seeds=seeds,
            output_dir=target,
            prompt=DEFAULT_PROMPT,
            expected_group_size=len(seeds),
            rollout_batch_size=1,
        )
        for artifact in artifacts:
            pending.append((raw, artifact))
    # Generate the fixed evaluation set before initializing/running reward
    # models. This prevents reward-side CUDA backend state from changing later
    # samples in the same evaluation set.
    for raw, artifact in pending:
        result = reward.score_paths(
            condition_id=raw.condition_id,
            prep_dir=raw.sample_dir,
            prediction_dir=artifact.seed_dir,
        )
        total = float(result["total_reward"])
        if not math.isfinite(total):
            raise FloatingPointError(f"non-finite evaluation reward for {raw.condition_id}")
        row = {
            "condition_id": raw.condition_id,
            "seed": artifact.seed,
            "policy_version": runtime.policy_version,
            "reward": result,
        }
        _jsonl(target / "rewards.jsonl", row)
        rows.append(row)
    totals = [float(row["reward"]["total_reward"]) for row in rows]
    leaf_rows = [_numeric_reward_leaves(row["reward"]) for row in rows]
    common_leaves = set.intersection(*(set(row) for row in leaf_rows)) if leaf_rows else set()
    component_statistics = {
        key: {
            "mean": float(np.mean([row[key] for row in leaf_rows])),
            "std": float(np.std([row[key] for row in leaf_rows])),
            "min": float(np.min([row[key] for row in leaf_rows])),
            "max": float(np.max([row[key] for row in leaf_rows])),
        }
        for key in sorted(common_leaves)
    }
    summary = {
        "tag": tag,
        "policy_version": runtime.policy_version,
        "num_rollouts": len(rows),
        "reward_mean": float(np.mean(totals)),
        "reward_std": float(np.std(totals)),
        "reward_min": float(np.min(totals)),
        "reward_max": float(np.max(totals)),
        "component_statistics": component_statistics,
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train(
    config: dict[str, Any],
    run_dir: Path,
    *,
    resume: str | None,
    max_steps: int | None,
    distributed: DistributedContext,
) -> None:
    from experiments.tempflow_video.reward_adapter import VideoRewardAdapter

    branching = bool(config["tempflow"]["trajectory_branching"])
    if distributed.enabled and not branching:
        raise NotImplementedError(
            "distributed runner currently supports TempFlow branching only, not ordinary video GRPO"
        )
    manifest = _make_manifest(config, run_dir, write_files=distributed.is_main)
    dataset = PrepConditionDataset(manifest)
    runtime = PersistentGeSimRuntime(config, device=distributed.device)
    policy = VideoPolicyAdapter(runtime)
    reference = ReferencePolicyAdapter(policy)
    trainer = TempFlowVideoTrainer(
        policy,
        reference,
        _optimizer_config(config),
        gradient_reducer=distributed.sum_tensors_ if distributed.enabled else None,
    )
    if resume:
        payload = load_tempflow_checkpoint(resume, map_location="cpu")
        _assert_resume_compatible(payload["config"], config)
        state = restore_tempflow_checkpoint(
            payload,
            policy=runtime.transformer,
            optimizer=trainer.optimizer,
            lr_scheduler=trainer.lr_scheduler,
        )
        trainer.load_state_dict(state)
        # Loading a checkpoint legitimately increments tensor version counters.
        # Establish the immutable-reference sentinel only after restore so later
        # in-place changes, rather than the restore itself, are what fail.
        reference = ReferencePolicyAdapter(policy)
        trainer.reference = reference
        runtime.set_policy_version(trainer.policy_version)

    reward = VideoRewardAdapter(config["reward"])
    sampler_class = TempFlowBranchSampler if branching else FlowGRPOVideoSampler
    sampler = sampler_class(
        runtime,
        eta=float(config["tempflow"]["eta"]),
        noise_aware_weighting=bool(config["tempflow"]["noise_aware_weighting"]),
        noise_weight_normalization=config["tempflow"].get("noise_weight_normalization", "schedule_mean"),
    )
    target_steps = int(max_steps or config.get("training", {}).get("max_optimizer_steps", config["max_optimizer_steps"]))
    branches = int(config["tempflow"]["branch_factor"])
    timesteps = [int(value) for value in config["tempflow"]["branch_timesteps"]]
    initial_seed = int(config["tempflow"]["initial_seed"])
    noise_seed_base = int(config["tempflow"]["branch_noise_seed_base"])
    checkpoint_every = int(config["training"].get("checkpoint_every", 1))
    if branches < distributed.world_size:
        raise ValueError(
            f"tempflow.branch_factor={branches} must be >= world_size={distributed.world_size}"
        )
    local_branch_ids = partition_indices(branches, distributed.world_size, distributed.rank)
    rank_output = (
        run_dir / "distributed" / f"rank_{distributed.rank:02d}"
        if distributed.enabled
        else run_dir
    )
    attempts = trainer.group_attempts
    max_attempts = max(target_steps * 4, target_steps + 2)
    while trainer.optimizer_step < target_steps and attempts < max_attempts:
        # One formal epoch is the full Cartesian condition x legal-timestep grid.
        condition_index = (attempts // len(timesteps)) % len(dataset) if branching else attempts % len(dataset)
        branch_timestep = timesteps[attempts % len(timesteps)] if branching else None
        raw = dataset[condition_index]
        prepared = runtime.prepare_condition(raw)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        common_sampling = {
            "output_dir": run_dir,
            "prompt": DEFAULT_PROMPT,
            "prompt_id": "default",
            "reward_config_sha256": reward.config_sha256,
            "video_length": int(config["rollout"]["total_frames"]),
        }
        if branching:
            global_seeds = [noise_seed_base + attempts * branches + offset for offset in range(branches)]
            seeds = [global_seeds[index] for index in local_branch_ids]
            group_dir, rollouts = sampler.sample_group(
                prepared,
                initial_seed=initial_seed,
                branch_timestep=branch_timestep,
                branch_noise_seeds=seeds,
                branch_ids=local_branch_ids,
                **{**common_sampling, "output_dir": rank_output},
            )
            prefix_hashes = distributed.gather_objects(rollouts[0].prefix_latent_sha256)
            if len(set(prefix_hashes)) != 1:
                raise RuntimeError(
                    "deterministic TempFlow prefix differs across ranks; refusing a mixed group"
                )
        else:
            initial_seeds = [initial_seed + attempts * branches + offset for offset in range(branches)]
            group_dir, rollouts = sampler.sample_group(
                prepared,
                initial_seeds=initial_seeds,
                transition_noise_seed_base=noise_seed_base + attempts * 1_000_000,
                group_sequence=attempts,
                **{**common_sampling, "output_dir": rank_output},
            )
        rewards = []
        valid_rollouts = []
        for rollout in rollouts:
            try:
                result = reward.score_rollout(rollout, prep_dir=raw.sample_dir)
                if config.get("reward_fusion", {}).get("mode") == "psnr_only_raw_db":
                    training_reward = float(result["geometry"]["metrics"]["balanced_psnr_db"])
                else:
                    training_reward = float(result["total_reward"])
                if not math.isfinite(training_reward):
                    raise FloatingPointError("terminal training reward is non-finite")
                rewards.append(training_reward)
                valid_rollouts.append(rollout)
            except Exception as exc:
                _jsonl(
                    rank_output / "invalid_rollouts.jsonl",
                    {"sample_id": rollout.sample_id, "error": repr(exc)},
                )
        minimum = int(config["reward"].get("min_valid_seeds_per_group", 2))
        attempts += 1
        trainer.group_attempts = attempts
        local_reward_rows = [
            (
                int(
                    rollout.branch_id
                    if isinstance(rollout, BranchRollout)
                    else rollout.rollout_id
                ),
                float(value),
            )
            for rollout, value in zip(valid_rollouts, rewards)
        ]
        gathered_reward_rows = distributed.gather_objects(local_reward_rows)
        if any(len(rows) == 0 for rows in gathered_reward_rows):
            raise RuntimeError("at least one rank has no valid PSNR branch; refusing desynchronized update")
        global_reward_rows = sorted(
            (row for rows in gathered_reward_rows for row in rows), key=lambda row: row[0]
        )
        global_ids = [row[0] for row in global_reward_rows]
        if len(global_ids) != len(set(global_ids)):
            raise RuntimeError("distributed branch shards contain duplicate global branch IDs")
        if len(global_reward_rows) < minimum:
            raise RuntimeError(
                f"valid reward branches {len(global_reward_rows)} < required {minimum}; refusing an invalid group"
            )
        advantages = standardize_group_rewards(
            [row[1] for row in global_reward_rows],
            epsilon=float(config["tempflow"].get("advantage_epsilon", 1.0e-6)),
            zero_std_threshold=float(config.get("reward_fusion", {}).get(
                "psnr_min_group_std_db", config["tempflow"].get("zero_std_threshold", 1.0e-8)
            )),
        )
        group_row = {
            "attempt": attempts,
            "condition_id": raw.condition_id,
            "branch_timestep": branch_timestep,
            "policy_version": runtime.policy_version,
            "elapsed_seconds_before_update": time.monotonic() - started,
            **advantages.metrics(),
        }
        if advantages.zero_std:
            group_row["optimizer_update_skipped"] = True
            if distributed.is_main:
                _jsonl(run_dir / "groups.jsonl", group_row)
            continue
        advantage_clip = float(config.get("reward_fusion", {}).get("advantage_clip", float("inf")))
        advantage_by_id = dict(zip(global_ids, advantages.advantages.tolist()))
        for rollout in valid_rollouts:
            member_id = (
                rollout.branch_id
                if isinstance(rollout, BranchRollout)
                else rollout.rollout_id
            )
            advantage = advantage_by_id[int(member_id)]
            rollout.advantage = float(max(-advantage_clip, min(advantage_clip, advantage)))
            (rollout.seed_dir / "rollout.json").write_text(
                json.dumps(rollout.metadata(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        record = trainer.update_group(
            valid_rollouts, global_group_size=len(global_reward_rows)
        )
        runtime.set_policy_version(record.policy_version)
        gathered_records = distributed.gather_objects(
            {"metrics": record.metrics, "local_count": len(valid_rollouts)}
        )
        combined_metrics = weighted_mean_dict(
            [item["metrics"] for item in gathered_records],
            [int(item["local_count"]) for item in gathered_records],
            shared_keys={
                "policy_grad_norm",
                "raw_kl_grad_norm",
                "weighted_kl_grad_norm",
                "total_grad_norm_before_clip",
                "total_grad_norm_after_clip",
                "learning_rate",
                "changed_trainable_parameter_tensors",
            },
        )
        group_row.update(
            {
                "optimizer_step": record.optimizer_step,
                "policy_version": record.policy_version,
                "metrics": combined_metrics,
                "world_size": distributed.world_size,
                "branches_per_rank": [len(rows) for rows in gathered_reward_rows],
            }
        )
        group_row["optimizer_update_skipped"] = False
        group_row["elapsed_seconds_total"] = time.monotonic() - started
        group_row["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        group_row["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if distributed.is_main:
            _jsonl(run_dir / "groups.jsonl", group_row)
        if trainer.optimizer_step % checkpoint_every == 0 or trainer.optimizer_step == target_steps:
            if distributed.is_main:
                save_tempflow_checkpoint(
                    run_dir / "checkpoints",
                    step=trainer.optimizer_step,
                    policy=runtime.transformer,
                    optimizer=trainer.optimizer,
                    lr_scheduler=trainer.lr_scheduler,
                    trainer_state=trainer.state_dict(),
                    config=config,
                )
            distributed.barrier()
        eval_every = int(config.get("evaluation", {}).get("every_optimizer_steps", 0) or 0)
        if (
            not bool(config.get("sampling", {}).get("smoke_only", False))
            and eval_every > 0
            and trainer.optimizer_step % eval_every == 0
        ):
            if distributed.is_main:
                summary = evaluate_policy(
                    runtime,
                    dataset,
                    reward,
                    run_dir=run_dir,
                    seeds=[
                        int(seed)
                        for seed in config["evaluation"].get(
                            "fixed_generation_seeds", [12345678]
                        )
                    ],
                    max_conditions=int(config["evaluation"].get("fixed_samples", 16)),
                    tag=f"policy_{runtime.policy_version:08d}",
                )
                _jsonl(run_dir / "evaluation_history.jsonl", summary)
            distributed.barrier()
        if distributed.is_main:
            print(json.dumps(group_row, ensure_ascii=False, default=str), flush=True)
    if trainer.optimizer_step < target_steps:
        raise RuntimeError(
            f"only completed {trainer.optimizer_step}/{target_steps} optimizer steps after {attempts} groups; "
            "reward variance was repeatedly zero"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="TempFlow-GRPO video adaptation runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--load-model-preflight", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    config = load_tempflow_config(args.config)
    os.environ["AWM_ASSET_ROOT"] = str(Path(config["model"]["checkpoint_root"]).parent)
    distributed = DistributedContext.initialize(
        int(config.get("distributed", {}).get("world_size", 1))
    )
    try:
        _seed_everything(int(config.get("seed", 42)))
        if not args.skip_preflight and distributed.is_main:
            report = run_preflight(config, load_model=args.load_model_preflight)
            report_path = Path(config["output_dir"]) / "preflight.json"
            write_preflight_report(report, report_path)
        distributed.barrier()
        if args.preflight_only:
            return
        local_run_dir = _new_run_dir(config, args.resume) if distributed.is_main else None
        run_dir = distributed.broadcast_path(local_run_dir)
        if distributed.is_main:
            dump_effective_config(config, run_dir / "effective_config.yaml")
        distributed.barrier()
        if config["experiment"]["mode"] == "base_eval":
            if distributed.is_main:
                from experiments.tempflow_video.reward_adapter import VideoRewardAdapter

                manifest = _make_manifest(config, run_dir)
                dataset = PrepConditionDataset(manifest)
                runtime = PersistentGeSimRuntime(config, device=distributed.device)
                reward = VideoRewardAdapter(config["reward"])
                summary = evaluate_policy(
                    runtime,
                    dataset,
                    reward,
                    run_dir=run_dir,
                    seeds=[int(seed) for seed in config["evaluation"]["fixed_generation_seeds"]],
                    max_conditions=int(config["evaluation"].get("fixed_samples", 16)),
                    tag="base_policy",
                )
                print(json.dumps(summary, ensure_ascii=False), flush=True)
            distributed.barrier()
        else:
            train(
                config,
                run_dir,
                resume=args.resume,
                max_steps=args.max_optimizer_steps,
                distributed=distributed,
            )
    finally:
        distributed.close()


if __name__ == "__main__":
    main()
