from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow both `python -m experiments.awm_coca.run_train` and direct execution
# from the repository root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.distributed as dist
import yaml
import numpy as np
from torch.utils.data import DataLoader

from experiments.awm_coca.async_reward import AsyncRewardRunner, RewardRequest
from experiments.awm_coca.checkpointing import load_checkpoint, restore_rng_state, save_checkpoint
from experiments.awm_coca.coca_credit import compute_credit
from experiments.awm_coca.condition_dataset import (
    PrepConditionDataset, build_manifest, collate_single_condition, write_manifest,
)
from experiments.awm_coca.condition_sampler import ResumableConditionSampler
from experiments.awm_coca.ema import ParameterEMA
from experiments.awm_coca.gesim_adapter import GeSimVelocityAdapter
from experiments.awm_coca.gesim_runtime import PersistentGeSimRuntime
from experiments.awm_coca.noise_schedule import build_training_noise_levels, validate_base_probabilities
from experiments.awm_coca.reward_runner import compute_head_reward
from experiments.awm_coca.trainer import AWMCoCATrainer, OptimizerConfig
from experiments.awm_coca.training_core import ProposalConfig
from experiments.awm_coca.training_data import load_fresh_rollout_group
from experiments.awm_coca.training_metric import JsonlMetricLogger
from experiments.awm_coca.wandb_monitor import WandbMonitor


def load_train_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("training config must be a YAML mapping")
    return config


def _repo_path(value: str | Path) -> str:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return str(expanded.resolve() if expanded.is_absolute() else (Path(_REPO_ROOT) / expanded).resolve())


def resolve_train_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve repo-relative and environment-variable paths without mutating the caller."""
    resolved = copy.deepcopy(config)
    resolved["output_dir"] = _repo_path(resolved["output_dir"])
    resolved["dataset"]["prep_root"] = _repo_path(resolved["dataset"]["prep_root"])
    resolved["model"]["gesim_config"] = _repo_path(resolved["model"]["gesim_config"])
    resolved["model"]["checkpoint_root"] = _repo_path(
        resolved["model"].get("checkpoint_root", "checkpoints")
    )
    reward = resolved["reward"]
    reward["gt_video_template"] = _repo_path(reward["gt_video_template"])
    reward["gt_video_templates"] = {
        camera: _repo_path(template)
        for camera, template in reward.get("gt_video_templates", {}).items()
    }
    return resolved


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    if args.prep_root:
        updated["dataset"]["prep_root"] = args.prep_root
    if args.gt_root:
        root = str(Path(args.gt_root))
        updated["reward"]["gt_video_template"] = str(Path(root) / "{condition_id}" / "head_29_frames.mp4")
        updated["reward"]["gt_video_templates"] = {
            camera: str(Path(root) / "{condition_id}" / f"{camera}_29_frames.mp4")
            for camera in ("head", "hand_left", "hand_right")
        }
    if args.output_dir:
        updated["output_dir"] = args.output_dir
    if args.gesim_config:
        updated["model"]["gesim_config"] = args.gesim_config
    if args.checkpoint_root:
        updated["model"]["checkpoint_root"] = args.checkpoint_root
    if args.rollout_retention:
        updated.setdefault("storage", {})["rollout_retention"] = args.rollout_retention
    if args.keep_consumed_rollouts:
        # 兼容旧命令；新命令使用 --rollout-retention all。
        updated.setdefault("storage", {})["rollout_retention"] = "all"
    scalar_overrides = (
        (args.group_size, "rollout", "group_size"),
        (args.max_optimizer_steps, None, "max_optimizer_steps"),
        (args.dataset_limit, "dataset", "limit"),
        (args.num_workers, "dataset", "num_workers"),
        (args.reward_workers, "reward", "workers"),
        (args.checkpoint_every, "checkpoint", "every_optimizer_steps"),
    )
    for value, section, key in scalar_overrides:
        if value is not None:
            target = updated if section is None else updated[section]
            target[key] = value
    if args.smoke_test:
        updated["rollout"]["group_size"] = 2
        updated["max_optimizer_steps"] = 1
        updated["dataset"]["limit"] = 1
        updated["dataset"]["num_workers"] = 0
        updated["dataset"]["pin_memory"] = False
        updated["reward"]["workers"] = 1
        updated["checkpoint"]["every_optimizer_steps"] = 0
        updated.setdefault("storage", {})["rollout_retention"] = "all"
        if not args.output_dir:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            updated["output_dir"] = f"outputs/awm_coca_smoke/{stamp}"
    return resolve_train_paths(updated)


def _checkpoint_asset(checkpoint_root: str | Path, configured_path: str | Path) -> Path:
    path = Path(os.path.expandvars(str(configured_path))).expanduser()
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    relative = Path(*parts[1:]) if parts and parts[0] == "checkpoints" else path
    return (Path(checkpoint_root) / relative).resolve()


def _configure_reward_backends(config: dict[str, Any]) -> None:
    from experiments.action_following import cowtracker_tracking, sam_tracking, yolo_detector

    checkpoint_root = Path(config["model"]["checkpoint_root"])
    sam_tracking.SAM3_CKPT = str(checkpoint_root / "sam3.pt")
    cowtracker_tracking.COWTRACKER_CKPT = str(
        checkpoint_root / "cowtracker" / "cowtracker_model.pth"
    )
    yolo_detector.YOLO_CKPT = str(checkpoint_root / "yoloworld-EWMBench-v0.1.pt")


def preflight(
    config: dict[str, Any], *, write_outputs: bool = True
) -> tuple[dict[str, Any], ResumableConditionSampler]:
    dataset = config["dataset"]
    manifest, invalid = build_manifest(
        dataset["prep_root"], include_samples=dataset.get("include_samples", ()),
        exclude_samples=dataset.get("exclude_samples", ()), limit=int(dataset.get("limit", 0)),
        validation_mode=dataset.get("validation_mode", "strict"),
    )
    output_dir = Path(config["output_dir"])
    if write_outputs:
        write_manifest(manifest, invalid, output_dir / "dataset")
    model_config = Path(config["model"]["gesim_config"])
    if not model_config.is_file():
        raise FileNotFoundError(f"GE-Sim config does not exist: {model_config}")
    with model_config.open("r", encoding="utf-8") as handle:
        ge_config = yaml.safe_load(handle)
    checkpoint_root = Path(config["model"]["checkpoint_root"])
    model_assets = [
        _checkpoint_asset(checkpoint_root, ge_config["pretrained_model_name_or_path"]),
        _checkpoint_asset(checkpoint_root, ge_config["tokenizer_pretrained_model_name_or_path"]),
        _checkpoint_asset(checkpoint_root, ge_config["diffusion_model"]["model_path"]),
    ]
    if ge_config.get("vae_path"):
        model_assets.append(_checkpoint_asset(checkpoint_root, ge_config["vae_path"]))
    missing_assets = [str(path) for path in model_assets if not path.exists()]
    if missing_assets:
        raise FileNotFoundError(
            f"missing {len(missing_assets)} GE-Sim model assets; first: {missing_assets[0]}"
        )
    reward_mode = config["reward"].get("mode", "action")
    if reward_mode in {"action", "joint"}:
        _configure_reward_backends(config)
        from experiments.action_following import cowtracker_tracking, sam_tracking, yolo_detector

        reward_assets = [
            Path(sam_tracking.SAM3_CKPT),
            Path(cowtracker_tracking.COWTRACKER_CKPT),
            Path(yolo_detector.YOLO_CKPT),
        ]
        missing_reward_assets = [str(path) for path in reward_assets if not path.is_file()]
        if missing_reward_assets:
            raise FileNotFoundError(
                f"missing {len(missing_reward_assets)} reward checkpoints; first: {missing_reward_assets[0]}"
            )
        if not Path(cowtracker_tracking.COWTRACKER_SRC).is_dir():
            raise FileNotFoundError(f"missing CoWTracker source: {cowtracker_tracking.COWTRACKER_SRC}")
        sam3_source = Path(sam_tracking.SAM3_SRC)
        if not sam3_source.is_dir():
            raise FileNotFoundError(f"missing SAM3 source: {sam3_source}")
        template = config["reward"]["gt_video_template"]
        missing_gt = [
            template.format(condition_id=entry["condition_id"])
            for entry in manifest["samples"]
            if not Path(template.format(condition_id=entry["condition_id"])).is_file()
        ]
        if missing_gt:
            raise FileNotFoundError(f"missing {len(missing_gt)} reward GT videos; first: {missing_gt[0]}")
    if reward_mode in {"geometry", "joint"}:
        if not bool(config["reward"].get("geometry_enabled", False)):
            raise ValueError(f"reward.mode={reward_mode} requires reward.geometry_enabled=true")
        templates = config["reward"].get("gt_video_templates", {})
        cameras = config["reward"].get("geometry_cameras", ())
        missing_geometry_gt = []
        for entry in manifest["samples"]:
            for camera in cameras:
                if camera not in templates:
                    missing_geometry_gt.append(f"missing template for camera={camera}")
                    continue
                path = templates[camera].format(condition_id=entry["condition_id"], camera=camera)
                if not Path(path).is_file():
                    missing_geometry_gt.append(path)
        if missing_geometry_gt:
            raise FileNotFoundError(
                f"missing {len(missing_geometry_gt)} geometry GT videos; first: {missing_geometry_gt[0]}"
            )
    sampler = ResumableConditionSampler(len(manifest["samples"]), seed=int(config.get("seed", 42)))
    return manifest, sampler


def _loader(config: dict[str, Any], manifest: dict[str, Any], sampler: ResumableConditionSampler) -> DataLoader:
    dataset_config = config["dataset"]
    workers = int(dataset_config.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "dataset": PrepConditionDataset(manifest), "batch_size": 1, "sampler": sampler,
        "collate_fn": collate_single_condition, "num_workers": workers,
        "pin_memory": bool(dataset_config.get("pin_memory", True)),
    }
    if workers:
        kwargs["persistent_workers"] = bool(dataset_config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(dataset_config.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


def _reward_function(config: dict[str, Any]):
    reward_config = config["reward"]
    template = reward_config["gt_video_template"]

    def evaluate(payload: dict[str, Any]) -> dict[str, Any]:
        seed_dir = Path(payload["seed_dir"])
        condition_id = payload["condition_id"]
        gt_path = template.format(condition_id=condition_id)
        geometry_cameras = reward_config.get("geometry_cameras", ("head", "hand_left", "hand_right"))
        gt_templates = reward_config.get("gt_video_templates", {})
        camera_videos = {
            camera: {
                "gt": gt_templates.get(camera, template).format(condition_id=condition_id, camera=camera),
                "pred": str(seed_dir / f"{camera}_color.mp4"),
            }
            for camera in geometry_cameras
        }
        return compute_head_reward(
            gt_path, str(seed_dir / "head_color.mp4"), prep_dir=payload["prep_dir"],
            max_frames=int(reward_config.get("max_frames", 29)),
            prompt=reward_config.get("segmentation_prompt", "robot arm"),
            confidence=float(reward_config.get("confidence", 0.1)),
            action_metric_weights=reward_config.get("action_metric_weights"),
            af_fdce_ate_norm_scale=float(reward_config.get("af_fdce_ate_norm_scale", 0.2)),
            fdce_scale=float(reward_config.get("fdce_scale", 10.0)),
            fdce_k=int(reward_config.get("fdce_k", 16)),
            fdce_visibility_threshold=float(reward_config.get("fdce_visibility_threshold", 0.5)),
            fdce_min_visible_fraction=float(reward_config.get("fdce_min_visible_fraction", 0.8)),
            fdce_min_common_frames=int(reward_config.get("fdce_min_common_frames", 1)),
            fdce_seed=int(reward_config.get("fdce_seed", 0)),
            reward_mode=reward_config.get("mode", "action"),
            geometry_enabled=bool(reward_config.get("geometry_enabled", False)),
            all_camera_videos=camera_videos,
            geometry_cameras=list(camera_videos),
            geometry_future_start=int(reward_config.get("geometry_future_start", 4)),
            geometry_future_end=int(reward_config.get("geometry_future_end", 28)),
            geometry_mean_weight=float(reward_config.get("geometry_mean_weight", 0.6)),
            geometry_worst_weight=float(reward_config.get("geometry_worst_weight", 0.4)),
            geometry_psnr_center_db=float(reward_config.get("geometry_psnr_center_db", 20.4)),
            geometry_psnr_temperature_db=float(reward_config.get("geometry_psnr_temperature_db", 1.8)),
            action_weight=float(reward_config.get("action_weight", 1.0)),
            geometry_weight=float(reward_config.get("geometry_weight", 1.0)),
        )
    return evaluate


def _score_group(config: dict[str, Any], runner: AsyncRewardRunner, group_dir: Path, raw: Any, version: int) -> None:
    futures = []
    for seed_dir in sorted(path for path in group_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
        rollout = json.loads((seed_dir / "rollout.json").read_text(encoding="utf-8"))
        seed = int(rollout["seed"])
        futures.append(runner.submit(RewardRequest(
            sample_id=seed_dir.name, condition_id=raw.condition_id, policy_version=version, seed=seed,
            payload={"seed_dir": str(seed_dir), "condition_id": raw.condition_id, "prep_dir": raw.sample_dir},
        )))
    results = runner.gather(
        futures, condition_id=raw.condition_id, policy_version=version,
        timeout=config["reward"].get("timeout_seconds"),
    )
    levels = int(config["proposal"]["num_training_noise_levels"])
    for result in results:
        seed_dir = group_dir / result.sample_id
        (seed_dir / "reward.json").write_text(json.dumps(result.reward, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            trajectory = torch.load(seed_dir / "trajectory.pt", map_location="cpu", weights_only=False)
        except TypeError:
            trajectory = torch.load(seed_dir / "trajectory.pt", map_location="cpu")
        credit = compute_credit(
            trajectory, float(result.reward["total_reward"]),
            window_size=int(config["proposal"].get("coca_window_size", 3)),
            num_training_noise_levels=levels,
            temperature=float(config["proposal"].get("temperature", 1.0)),
        )
        (seed_dir / "credit.json").write_text(json.dumps(credit, ensure_ascii=False, indent=2), encoding="utf-8")


def _compact_rollout_rows(group_dir: Path) -> list[dict[str, Any]]:
    group = json.loads((group_dir / "group.json").read_text(encoding="utf-8"))
    rows = []
    for seed_dir in sorted(path for path in group_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
        reward = json.loads((seed_dir / "reward.json").read_text(encoding="utf-8"))
        credit = json.loads((seed_dir / "credit.json").read_text(encoding="utf-8"))
        rows.append({
            "group_id": group["group_id"],
            "condition_id": group["condition_id"],
            "policy_version": group["policy_version"],
            "sample_id": seed_dir.name,
            "reward": reward,
            "credit": {key: value for key, value in credit.items() if key != "step_rows"},
        })
    return rows


def _remove_consumed_group(group_dir: Path, output_dir: Path) -> None:
    target = group_dir.resolve()
    rollout_root = (output_dir / "rollouts").resolve()
    if target.parent != rollout_root or not (target / "group.json").is_file():
        raise ValueError(f"refusing to remove unexpected rollout path: {target}")
    shutil.rmtree(target)


def _retain_consumed_group(group_dir: Path, output_dir: Path, retention: str) -> None:
    """训练消费成功后按策略清理中间张量；只操作本次 output/rollouts 下的合法 group。"""
    target = group_dir.resolve()
    rollout_root = (output_dir / "rollouts").resolve()
    if target.parent != rollout_root or not (target / "group.json").is_file():
        raise ValueError(f"refusing to prune unexpected rollout path: {target}")
    retention = str(retention).lower()
    if retention == "all":
        return
    if retention == "none":
        _remove_consumed_group(target, output_dir)
        return
    if retention != "videos":
        raise ValueError(f"unknown rollout retention: {retention}; expected all/videos/none")

    # reward、credit、rollout/group 元数据和所有 MP4 均保留；仅删掉训练已经消费的大张量。
    tensor_paths = [target / "condition.pt"]
    for seed_dir in target.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            tensor_paths.extend((seed_dir / "trajectory.pt", seed_dir / "final_future_latent.pt"))
    for path in tensor_paths:
        if path.is_file():
            path.unlink()


def _distributed_info() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _gather_per_rank(value: Any) -> list[Any]:
    _, world_size = _distributed_info()
    if world_size == 1:
        return [value]
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


def _reward_pairs(group_dir: Path) -> list[tuple[int, float]]:
    pairs = []
    for seed_dir in sorted(path for path in group_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
        rollout = json.loads((seed_dir / "rollout.json").read_text(encoding="utf-8"))
        reward = json.loads((seed_dir / "reward.json").read_text(encoding="utf-8"))
        pairs.append((int(rollout["seed"]), float(reward["total_reward"])))
    return pairs


def _global_group_metrics(per_rank: list[dict[str, Any]]) -> dict[str, Any]:
    first = per_rank[0]
    versions = {(row["optimizer_step"], row["policy_version"], row["optimizer_stepped"]) for row in per_rank}
    if len(versions) != 1:
        raise RuntimeError(f"distributed trainer state diverged across ranks: {sorted(versions)}")
    grad_norms = [float(row["grad_norm"]) for row in per_rank if row["grad_norm"] is not None]
    return {
        **first,
        "loss": float(sum(float(row["loss"]) for row in per_rank) / len(per_rank)),
        "grad_norm": None if not grad_norms else float(sum(grad_norms) / len(grad_norms)),
        "samples": [sample for row in per_rank for sample in row["samples"]],
        "world_size": len(per_rank),
    }


def _restore_adapter(policy: torch.nn.Module, checkpoint: Path) -> None:
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    state = load_peft_weights(str(checkpoint / "policy_lora"), device="cpu")
    result = set_peft_model_state_dict(policy, state, adapter_name="default")
    if getattr(result, "unexpected_keys", None):
        raise ValueError(f"unexpected LoRA checkpoint keys: {result.unexpected_keys}")


def train(config: dict[str, Any], *, resume: str | None = None, device: str = "cuda") -> None:
    config = resolve_train_paths(config)
    _configure_reward_backends(config)
    rank, world_size = _distributed_info()
    initialization_seed = int(config.get("seed", 42))
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    global_group_size = int(config["rollout"]["group_size"])
    if not bool(config["rollout"].get("fresh_on_policy", True)):
        raise ValueError("this trainer only supports fresh on-policy rollouts")
    if global_group_size % world_size:
        raise ValueError(
            f"rollout.group_size={global_group_size} must be divisible by world_size={world_size}"
        )
    local_group_size = global_group_size // world_size
    if local_group_size < 2:
        raise ValueError(
            "each rank needs at least two local rollouts; increase rollout.group_size "
            f"to at least {2 * world_size}"
        )
    manifest, sampler = preflight(config, write_outputs=rank == 0)
    if world_size > 1:
        dist.barrier()
    runtime = PersistentGeSimRuntime(config, device=device)
    # The GE-Sim rollout pipeline re-derives its own sigma schedule inside
    # infer() (gesim_pipeline.py uses sigmas=linspace(0,1,N), then applies the
    # configured invert_sigmas/shift). Replicate that exact call here so the AWM
    # training noise levels fall in the same flow-time range as the reverse
    # trajectory. Using the scheduler's default (EDM sigma_max=80) schedule
    # instead puts ~all levels near pure noise (flow_time ~0.92-0.99), which
    # decouples the CoCA bins (spanning t in [0,0.5]) from the levels' noise.
    n_rollout = int(config["rollout"]["reverse_denoise_steps"])
    rollout_sigmas = torch.linspace(0, 1, n_rollout, dtype=torch.float64)
    runtime.scheduler.set_timesteps(sigmas=rollout_sigmas, device=runtime.device)
    noise_levels = build_training_noise_levels(runtime.scheduler, int(config["proposal"]["num_training_noise_levels"]))
    proposal = ProposalConfig(
        noise_levels=tuple(level.flow_time for level in noise_levels),
        eta=float(config["proposal"]["eta"]), temperature=float(config["proposal"].get("temperature", 1.0)),
        base_probabilities=validate_base_probabilities(
            config["proposal"]["base_probabilities"], len(noise_levels)
        ),
        importance_clipping=config["proposal"].get("importance_clipping"),
    )
    optimizer_config = config["optimizer"]
    ema = None
    if bool(config["ema"].get("enabled", True)):
        ema = ParameterEMA(runtime.transformer.named_parameters(), decay=float(config["ema"]["decay"]))
    trainer = AWMCoCATrainer(
        GeSimVelocityAdapter(runtime.transformer), proposal,
        OptimizerConfig(
            learning_rate=float(optimizer_config["learning_rate"]), betas=tuple(optimizer_config["betas"]),
            epsilon=float(optimizer_config["epsilon"]), weight_decay=float(optimizer_config["weight_decay"]),
            max_grad_norm=float(optimizer_config["max_grad_norm"]),
            beta=float(optimizer_config["reference_kl_beta"]),
            gradient_accumulation_steps=int(optimizer_config["gradient_accumulation_steps"]),
            warmup_steps=int(optimizer_config["warmup_steps"]),
        ), ema=ema,
    )
    if resume:
        checkpoint = load_checkpoint(resume, map_location="cpu")
        _restore_adapter(runtime.transformer, checkpoint["path"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        trainer.load_state_dict(checkpoint["trainer_state"])
        if ema is not None:
            ema.load_state_dict(checkpoint["ema"])
        sampler.load_state_dict(checkpoint["sampler"])
        restore_rng_state(checkpoint["rng"])
        runtime.set_policy_version(trainer.policy_version)
    if world_size > 1:
        for parameter in runtime.transformer.parameters():
            if parameter.requires_grad:
                dist.broadcast(parameter.data, src=0)
        dist.barrier()
    output = Path(config["output_dir"])
    logger = JsonlMetricLogger(output / "metrics" / "train.jsonl") if rank == 0 else None
    rollout_logger = JsonlMetricLogger(output / "metrics" / "rollouts.jsonl") if rank == 0 else None
    max_steps = int(config["max_optimizer_steps"])
    base_seed = int(config.get("seed", 42))
    group_counter = (
        trainer.optimizer_step * int(optimizer_config["gradient_accumulation_steps"])
        + trainer.accumulation_step
    )
    with (
        WandbMonitor(config, output, enabled=rank == 0) as monitor,
        AsyncRewardRunner(
            _reward_function(config), workers=int(config["reward"].get("workers", 1))
        ) as rewards,
    ):
        while trainer.optimizer_step < max_steps:
            for raw in _loader(config, manifest, sampler):
                version = trainer.policy_version
                runtime.set_policy_version(version)
                prepared = runtime.prepare_condition(raw)
                first_seed = base_seed + group_counter * global_group_size
                global_group_id = f"{raw.condition_id}_policy_{version:08d}_seed_{first_seed}"
                all_seeds = [first_seed + offset for offset in range(global_group_size)]
                local_seeds = all_seeds[rank * local_group_size : (rank + 1) * local_group_size]
                group_dir = runtime.rollout_group(
                    prepared, seeds=local_seeds, output_dir=output,
                    expected_group_size=local_group_size,
                )
                _score_group(config, rewards, group_dir, raw, version)
                local_rows = _compact_rollout_rows(group_dir)
                for row in local_rows:
                    row["global_group_id"] = global_group_id
                    row["rank"] = rank
                gathered_rows = _gather_per_rank(local_rows)
                global_rollout_rows = [item for rows in gathered_rows for item in rows]
                if rollout_logger is not None:
                    for row in global_rollout_rows:
                        rollout_logger.write(row)
                gathered_rewards = _gather_per_rank(_reward_pairs(group_dir))
                ordered_rewards = sorted(
                    (item for rows in gathered_rewards for item in rows), key=lambda item: item[0]
                )
                if [seed for seed, _ in ordered_rewards] != all_seeds:
                    raise RuntimeError("distributed rollout seed gather is incomplete or duplicated")
                global_rewards = [reward for _, reward in ordered_rewards]
                _, samples = load_fresh_rollout_group(
                    group_dir, device=runtime.device, expected_group_size=local_group_size,
                    expected_policy_version=version,
                    global_rewards=global_rewards,
                )
                local_metrics = trainer.update_group(samples, group_id=global_group_id)
                metrics = _global_group_metrics(_gather_per_rank(local_metrics))
                train_row = {"condition_id": raw.condition_id, "group_id": global_group_id, **metrics}
                if logger is not None:
                    logger.write(train_row)
                    monitor.log_group(
                        train_row, global_rollout_rows, group_step=group_counter + 1
                    )
                group_counter += 1
                if metrics["optimizer_stepped"]:
                    runtime.set_policy_version(trainer.policy_version)
                    every = int(config["checkpoint"]["every_optimizer_steps"])
                    if rank == 0 and every > 0 and trainer.optimizer_step % every == 0:
                        save_checkpoint(
                            output / "checkpoints", step=trainer.optimizer_step, policy=runtime.transformer,
                            optimizer=trainer.optimizer, lr_scheduler=trainer.lr_scheduler,
                            ema_state={} if ema is None else ema.state_dict(), sampler_state=sampler.state_dict(),
                            trainer_state=trainer.state_dict(), config=config,
                        )
                if world_size > 1:
                    dist.barrier()
                retention = config.get("storage", {}).get("rollout_retention")
                if retention is None:
                    # 兼容旧配置：true=all，false=none。
                    keep_legacy = bool(config.get("storage", {}).get("keep_consumed_rollouts", True))
                    retention = "all" if keep_legacy else "none"
                _retain_consumed_group(group_dir, output, str(retention))
                del samples, prepared
                if trainer.optimizer_step >= max_steps:
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="AWM-CoCA persistent single-chunk video RL training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="one sample, group=2, one optimizer step, no checkpoint")
    parser.add_argument("--prep-root")
    parser.add_argument("--gt-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--gesim-config")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--reward-workers", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--keep-consumed-rollouts", action="store_true")
    parser.add_argument(
        "--rollout-retention",
        choices=("all", "videos", "none"),
        help="已消费 rollout 的保留策略；正式训练推荐 videos",
    )
    parser.add_argument("--print-effective-config", action="store_true")
    parser.add_argument(
        "--effective-config-output",
        help="将生效配置直接写入 YAML 文件，避免第三方库 stdout 日志污染文件",
    )
    args = parser.parse_args()
    env_rank = int(os.environ.get("RANK", "0"))
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config = apply_cli_overrides(load_train_config(args.config), args)
    if args.print_effective_config:
        if env_rank == 0:
            payload = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
            if args.effective_config_output:
                output_path = Path(args.effective_config_output).expanduser().resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(payload, encoding="utf-8")
                print(f"[INFO] 生效配置已写入: {output_path}")
            else:
                print(payload)
        return
    if args.preflight_only:
        manifest, _ = preflight(config, write_outputs=env_rank == 0)
        if env_rank == 0:
            print(json.dumps({"manifest_sha256": manifest["sha256"], "num_samples": manifest["num_samples"]}))
        return
    initialized_here = False
    train_device = args.device
    if env_world_size > 1:
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            parser.error("multi-process training requires CUDA and one visible GPU per rank")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        initialized_here = True
        train_device = f"cuda:{local_rank}"
    try:
        train(config, resume=args.resume, device=train_device)
    finally:
        if initialized_here:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
