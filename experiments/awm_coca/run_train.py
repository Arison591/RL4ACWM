from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timedelta
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
from tqdm.auto import tqdm

from experiments.awm_coca.async_reward import AsyncRewardRunner, RewardRequest
from experiments.awm_coca.checkpointing import (
    AsyncCheckpointWriter,
    build_checkpoint_snapshot,
    load_checkpoint,
    restore_rng_state,
)
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
    if "eval" in resolved:
        eval_settings = resolved["eval"]
        if eval_settings.get("enabled", False):
            if not eval_settings.get("prep_root"):
                raise ValueError("eval.enabled requires eval.prep_root")
            eval_settings["prep_root"] = _repo_path(eval_settings["prep_root"])
            if eval_settings.get("gt_video_template"):
                eval_settings["gt_video_template"] = _repo_path(eval_settings["gt_video_template"])
            if eval_settings.get("gt_video_templates"):
                eval_settings["gt_video_templates"] = {
                    camera: _repo_path(template)
                    for camera, template in eval_settings["gt_video_templates"].items()
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
    eval_settings = updated.setdefault("eval", {})
    eval_prep_root = getattr(args, "eval_prep_root", None)
    if eval_prep_root:
        eval_settings["prep_root"] = eval_prep_root
        eval_settings["enabled"] = True
    eval_scalar_overrides = (
        (getattr(args, "eval_every_group_steps", None), "every_group_steps"),
        (getattr(args, "eval_max_conditions", None), "max_conditions"),
        (getattr(args, "eval_seeds_per_condition", None), "seeds_per_condition"),
        (getattr(args, "eval_rollout_batch_size", None), "rollout_batch_size"),
        (getattr(args, "eval_seed", None), "seed"),
    )
    for value, key in eval_scalar_overrides:
        if value is not None:
            eval_settings[key] = value
    if args.smoke_test:
        updated["rollout"]["group_size"] = 2
        # A group=2 smoke run can never satisfy the production threshold of 8;
        # keep the leave-one-out minimum while allowing its single update to run.
        updated["reward"]["min_valid_seeds_per_group"] = 2
        updated["max_optimizer_steps"] = 1
        updated["dataset"]["limit"] = 1
        updated["dataset"]["num_workers"] = 0
        updated["dataset"]["pin_memory"] = False
        # 冒烟只跑 1 个训练 group，不付完整测试集 rollout+reward 的开销。
        updated.setdefault("eval", {})["enabled"] = False
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


def _eval_gt_templates(config: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Resolve the eval reward GT templates (eval override or training fallback)."""
    eval_settings = config.get("eval", {})
    head_template = eval_settings.get("gt_video_template")
    camera_templates = eval_settings.get("gt_video_templates")
    if head_template is None:
        head_template = config["reward"]["gt_video_template"]
    if camera_templates is None:
        camera_templates = config["reward"].get("gt_video_templates", {})
    return head_template, camera_templates


def _preflight_eval(config: dict[str, Any]) -> dict[str, Any] | None:
    """Build and validate the held-out eval manifest; None when eval is disabled.

    Mirrors the training-manifest reward GT checks so `--preflight-only` catches
    a broken eval set before any rollout runs.
    """
    eval_settings = config.get("eval", {})
    if not eval_settings.get("enabled", False):
        return None
    if int(eval_settings.get("every_group_steps", 10)) < 1:
        raise ValueError("eval.every_group_steps must be >= 1")
    seeds_per_condition = int(eval_settings.get("seeds_per_condition", 2))
    rollout_batch_size = int(eval_settings.get("rollout_batch_size", 1))
    if seeds_per_condition < 1:
        raise ValueError("eval.seeds_per_condition must be >= 1")
    if rollout_batch_size < 1 or seeds_per_condition % rollout_batch_size:
        raise ValueError(
            "eval.rollout_batch_size must be positive and divide "
            f"eval.seeds_per_condition ({seeds_per_condition})"
        )
    eval_manifest, _ = build_manifest(
        eval_settings["prep_root"], validation_mode=eval_settings.get("validation_mode", "strict"),
    )
    head_template, camera_templates = _eval_gt_templates(config)
    max_conditions = eval_settings.get("max_conditions")
    entries = sorted(eval_manifest["samples"], key=lambda entry: entry["condition_id"])
    if max_conditions is not None:
        entries = entries[: int(max_conditions)]
    missing_head = [
        head_template.format(condition_id=entry["condition_id"])
        for entry in entries
        if not Path(head_template.format(condition_id=entry["condition_id"])).is_file()
    ]
    if missing_head:
        raise FileNotFoundError(
            f"missing {len(missing_head)} eval head GT videos; first: {missing_head[0]}"
        )
    reward_mode = config["reward"].get("mode", "action")
    if reward_mode in {"geometry", "joint"}:
        cameras = config["reward"].get("geometry_cameras", ())
        missing_geometry = []
        for entry in entries:
            for camera in cameras:
                if camera not in camera_templates:
                    missing_geometry.append(f"missing eval template for camera={camera}")
                    continue
                path = camera_templates[camera].format(
                    condition_id=entry["condition_id"], camera=camera
                )
                if not Path(path).is_file():
                    missing_geometry.append(path)
        if missing_geometry:
            raise FileNotFoundError(
                f"missing {len(missing_geometry)} eval geometry GT videos; first: {missing_geometry[0]}"
            )
    return eval_manifest


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


def _score_group(
    config: dict[str, Any], runner: AsyncRewardRunner, group_dir: Path, raw: Any, version: int,
    *,
    artifacts: dict[int, Any] | None = None,
) -> None:
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
        total_reward = result.reward.get("total_reward")
        if total_reward is None or not result.reward.get("valid", True):
            # SAM3/跟踪失败等导致该 seed 奖励无效：保留 reward.json，不计算 credit，
            # 训练阶段会过滤掉该 seed（无效 seed < 足够数量时跳过整个 task）。
            (seed_dir / "credit.json").write_text(json.dumps(
                {"valid": False, "noise_rows": [], "error": result.reward.get("error")},
                ensure_ascii=False, indent=2,
            ), encoding="utf-8")
            continue
        if artifacts is not None:
            artifact = artifacts.get(result.seed)
            if artifact is None:
                raise ValueError(f"missing in-memory rollout artifact for seed {result.seed}")
            trajectory = {"chunks": [artifact.trajectory], "num_chunks": 1}
        else:
            try:
                trajectory = torch.load(seed_dir / "trajectory.pt", map_location="cpu", weights_only=False)
            except TypeError:
                trajectory = torch.load(seed_dir / "trajectory.pt", map_location="cpu")
        credit = compute_credit(
            trajectory, float(total_reward),
            window_size=int(config["proposal"].get("coca_window_size", 3)),
            num_training_noise_levels=levels,
            temperature=float(config["proposal"].get("temperature", 1.0)),
            credit_source=str(config["proposal"].get("credit_source", "predicted_x0")),
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


def _distributed_timeout() -> timedelta:
    """Allow slow, rank-skewed rollout/reward stages before collectives.

    A rank that finishes reward evaluation early waits in the next gather while
    slower ranks are still running SAM3/CoWTracker.  PyTorch's 10-minute NCCL
    timeout is shorter than a valid reward stage on the target machine.
    """
    seconds = int(os.environ.get("DIST_TIMEOUT_SECONDS", "7200"))
    if seconds <= 0:
        raise ValueError("DIST_TIMEOUT_SECONDS must be positive")
    return timedelta(seconds=seconds)


def _gather_per_rank(value: Any) -> list[Any]:
    _, world_size = _distributed_info()
    if world_size == 1:
        return [value]
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


def _prepare_eval_rollout_root(output: Path, *, rank: int, world_size: int) -> None:
    """Clean stale eval groups once, then release every rank together.

    The eval output is shared by all local ranks.  Letting every rank remove
    stale directories races with faster ranks already writing their new groups,
    which can delete a freshly-created ``group.json`` mid-evaluation.
    """
    eval_rollout_root = output / "eval" / "rollouts"
    if rank == 0 and eval_rollout_root.is_dir():
        for stale in eval_rollout_root.iterdir():
            if stale.is_dir():
                shutil.rmtree(stale)
    if world_size > 1:
        dist.barrier()


def _run_eval(
    config: dict[str, Any],
    eval_config: dict[str, Any],
    runtime: PersistentGeSimRuntime,
    eval_rewards: AsyncRewardRunner,
    output: Path,
    eval_manifest: dict[str, Any],
    *,
    version: int,
    group_step: int,
    eval_logger: JsonlMetricLogger | None,
    monitor: WandbMonitor | None,
) -> None:
    """Roll out the current policy on the held-out eval set and log aggregated reward.

    Inference only — no policy update, no training-sampler consumption.  Each rank
    rolls out its share of the (deterministically selected) eval conditions, then
    rewards are gathered across ranks and logged by rank 0 under eval/*.
    """
    eval_settings = config["eval"]
    rank, world_size = _distributed_info()
    max_conditions = eval_settings.get("max_conditions")
    seeds_per_condition = max(1, int(eval_settings.get("seeds_per_condition", 2)))
    rollout_batch_size = int(eval_settings.get("rollout_batch_size", 1))
    eval_seed = int(eval_settings.get("seed", 12345))

    # 共享目录只由 rank 0 清理；barrier 后所有 rank 再开始写各自的新 group。
    _prepare_eval_rollout_root(output, rank=rank, world_size=world_size)

    runtime.set_policy_version(version)
    dataset = PrepConditionDataset(eval_manifest)
    entries = sorted(eval_manifest["samples"], key=lambda entry: entry["condition_id"])
    if max_conditions is not None:
        entries = entries[: int(max_conditions)]
    local_entries = entries[rank::world_size]

    rows: list[dict[str, Any]] = []
    for entry in local_entries:
        raw = dataset[int(entry["index"])]
        prepared = runtime.prepare_condition(raw)
        seeds = [eval_seed + offset for offset in range(seeds_per_condition)]
        group_dir, artifacts = runtime.rollout_group(
            prepared, seeds=seeds, output_dir=output / "eval",
            expected_group_size=seeds_per_condition, rollout_batch_size=rollout_batch_size,
        )
        # rollout_group 不落盘 trajectory.pt，credit 必须走内存 artifact。
        artifacts_by_seed = {artifact.seed: artifact for artifact in artifacts}
        _score_group(eval_config, eval_rewards, group_dir, raw, version, artifacts=artifacts_by_seed)
        for row in _compact_rollout_rows(group_dir):
            row["group_step"] = group_step
            rows.append(row)
        _remove_consumed_group(group_dir, output / "eval")

    gathered = _gather_per_rank(rows)
    global_rows = [item for group in gathered for item in group]
    totals = [
        float(row["reward"]["total_reward"])
        for row in global_rows
        if row["reward"].get("total_reward") is not None and row["reward"].get("valid", True)
    ]
    action_rewards = [
        float(row["reward"]["action_reward"]) for row in global_rows
        if row["reward"].get("action_reward") is not None and row["reward"].get("valid", True)
    ]
    geometry_rewards = [
        float(row["reward"]["geometry_reward"]) for row in global_rows
        if row["reward"].get("geometry_reward") is not None and row["reward"].get("valid", True)
    ]

    def _mean(values: list[float]) -> float | None:
        return None if not values else float(sum(values) / len(values))

    mean_reward = _mean(totals)
    valid_fraction = float(len(totals) / len(global_rows)) if global_rows else None
    if rank == 0:
        row = {
            "group_step": group_step,
            "policy_version": version,
            "n_conditions": len(entries),
            "n_seeds": len(global_rows),
            "n_valid_seeds": len(totals),
            "mean_reward": mean_reward,
            "mean_action_reward": _mean(action_rewards),
            "mean_geometry_reward": _mean(geometry_rewards),
            "valid_fraction": valid_fraction,
            "condition_ids": [entry["condition_id"] for entry in entries],
        }
        if eval_logger is not None:
            eval_logger.write(row)
        if monitor is not None:
            eval_payload = {
                "eval/reward_total": mean_reward,
                "eval/reward_action": _mean(action_rewards),
                "eval/reward_geometry": _mean(geometry_rewards),
                "eval/valid_fraction": valid_fraction,
                "eval/policy_version": version,
                "eval/n_valid_seeds": len(totals),
            }
            monitor.log_eval(
                {key: value for key, value in eval_payload.items() if value is not None},
                group_step=group_step,
            )


def _assert_group_alignment(condition_id: str, policy_version: int, group_counter: int) -> None:
    """Fail early if ranks are about to train different logical groups."""
    local = (str(condition_id), int(policy_version), int(group_counter))
    gathered = _gather_per_rank(local)
    if any(item != gathered[0] for item in gathered[1:]):
        raise RuntimeError(f"distributed group alignment mismatch: {gathered}")


def _sampler_state_after_groups(
    sampler: ResumableConditionSampler, completed_groups: int
) -> dict[str, Any]:
    """Return the exact next-condition state, independent of loader prefetch."""
    if completed_groups < 0:
        raise ValueError("completed_groups must be non-negative")
    per_epoch = len(sampler)
    epoch, position = divmod(int(completed_groups), per_epoch)
    state = sampler.state_dict()
    state["epoch"] = epoch
    state["position"] = position
    return state


def _reward_pairs(group_dir: Path) -> list[tuple[int, float]]:
    pairs = []
    for seed_dir in sorted(path for path in group_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
        rollout = json.loads((seed_dir / "rollout.json").read_text(encoding="utf-8"))
        reward = json.loads((seed_dir / "reward.json").read_text(encoding="utf-8"))
        if reward.get("total_reward") is None or not reward.get("valid", True):
            continue  # 该 seed 奖励无效（跟踪失败等），训练时跳过
        pairs.append((int(rollout["seed"]), float(reward["total_reward"])))
    return pairs


def _group_skip_reason(
    gathered_rewards: list[list[tuple[int, float]]],
    *,
    min_valid_seeds: int,
) -> str | None:
    """Return the rank-synchronous reason for skipping a rollout group.

    Every rank calls this after ``all_gather_object``, so every process sees the
    same per-rank counts and takes the same train/skip branch.  In particular,
    a globally large enough group is still skipped when one rank has no valid
    local sample; otherwise that rank cannot enter DDP backward while its peers
    do, which eventually deadlocks in a collective.
    """
    local_valid_counts = [len(rows) for rows in gathered_rewards]
    total_valid = sum(local_valid_counts)
    if total_valid < min_valid_seeds:
        return f"only {total_valid} valid seeds (need >= {min_valid_seeds})"
    empty_ranks = [rank for rank, count in enumerate(local_valid_counts) if count == 0]
    if empty_ranks:
        return f"no valid seed on rank(s) {empty_ranks}"
    return None


def _global_group_metrics(per_rank: list[dict[str, Any]]) -> dict[str, Any]:
    first = per_rank[0]
    versions = {(row["optimizer_step"], row["policy_version"], row["optimizer_stepped"]) for row in per_rank}
    if len(versions) != 1:
        raise RuntimeError(f"distributed trainer state diverged across ranks: {sorted(versions)}")
    def _averaged(key: str) -> float | None:
        values = [float(row[key]) for row in per_rank if row.get(key) is not None]
        return None if not values else float(sum(values) / len(values))

    return {
        **first,
        "loss": float(sum(float(row["loss"]) for row in per_rank) / len(per_rank)),
        "grad_norm": _averaged("grad_norm"),
        "fm_grad_norm": _averaged("fm_grad_norm"),
        "kl_grad_norm": _averaged("kl_grad_norm"),
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
    rollout_batch_size = int(config["rollout"].get("rollout_batch_size", 1))
    if rollout_batch_size <= 0 or local_group_size % rollout_batch_size:
        raise ValueError(
            f"rollout_batch_size={rollout_batch_size} must divide per-rank group size "
            f"{local_group_size} (global={global_group_size}, world_size={world_size})"
        )
    manifest, sampler = preflight(config, write_outputs=rank == 0)
    if world_size > 1:
        dist.barrier()
    eval_manifest = _preflight_eval(config)
    eval_config = None
    if eval_manifest is not None:
        # 独立测试集：评估奖励沿用训练 reward 口径，但 GT 模板允许 eval 配置覆盖。
        eval_config = copy.deepcopy(config)
        head_template, camera_templates = _eval_gt_templates(config)
        eval_config["reward"]["gt_video_template"] = head_template
        eval_config["reward"]["gt_video_templates"] = dict(camera_templates)
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
            log_term_grad_norm=bool(optimizer_config.get("log_term_grad_norm", True)),
        ), ema=ema,
    )
    restored_group_counter: int | None = None
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
        if "group_counter" in checkpoint["trainer_state"]:
            restored_group_counter = int(checkpoint["trainer_state"]["group_counter"])
    if world_size > 1:
        for parameter in runtime.transformer.parameters():
            if parameter.requires_grad:
                dist.broadcast(parameter.data, src=0)
        dist.barrier()
    output = Path(config["output_dir"])
    logger = JsonlMetricLogger(output / "metrics" / "train.jsonl") if rank == 0 else None
    rollout_logger = JsonlMetricLogger(output / "metrics" / "rollouts.jsonl") if rank == 0 else None
    eval_logger = JsonlMetricLogger(output / "metrics" / "eval.jsonl") if rank == 0 else None
    eval_every = (
        int(config["eval"].get("every_group_steps", 10)) if eval_manifest is not None else 0
    )
    max_steps = int(config["max_optimizer_steps"])
    conditions_per_epoch = len(sampler)
    accumulation_steps = int(optimizer_config["gradient_accumulation_steps"])
    estimated_total_epochs = max_steps * accumulation_steps / conditions_per_epoch
    base_seed = int(config.get("seed", 42))
    group_counter = (
        restored_group_counter
        if restored_group_counter is not None
        else trainer.optimizer_step * int(optimizer_config["gradient_accumulation_steps"])
        + trainer.accumulation_step
    )
    progress = tqdm(
        total=max_steps,
        initial=trainer.optimizer_step,
        desc="AWM-CoCA 训练",
        unit="step",
        dynamic_ncols=True,
        disable=rank != 0,
    )
    if rank == 0:
        progress.write(
            f"[INFO] 训练长度：{max_steps} 次 optimizer update；"
            f"每轮 {conditions_per_epoch} 个 condition；约 {estimated_total_epochs:.2f} 个 epoch"
        )
    reward_cuda_device = None
    if runtime.device.type == "cuda":
        reward_cuda_device = runtime.device.index
        if reward_cuda_device is None:
            reward_cuda_device = torch.cuda.current_device()
        progress.write(
            f"[INFO][rank {rank}] 训练与 reward worker 绑定 logical cuda:{reward_cuda_device}"
        )
    eval_reward_ctx = (
        AsyncRewardRunner(
            _reward_function(eval_config),
            workers=int(config["reward"].get("workers", 1)),
            cuda_device=reward_cuda_device,
        )
        if eval_config is not None
        else contextlib.nullcontext(None)
    )
    with (
        progress,
        WandbMonitor(config, output, enabled=rank == 0) as monitor,
        AsyncRewardRunner(
            _reward_function(config),
            workers=int(config["reward"].get("workers", 1)),
            cuda_device=reward_cuda_device,
        ) as rewards,
        eval_reward_ctx as eval_rewards,
    ):
        checkpoint_writer = AsyncCheckpointWriter() if rank == 0 else None
        while trainer.optimizer_step < max_steps:
            for raw in _loader(config, manifest, sampler):
                # 每 eval_every 个 group step 在独立测试集上评估一次（group 0 先出基线）。
                # 只做 rollout+reward，不更新策略、不推进训练 sampler；group_counter 全局一致，
                # 所有 rank 同时进入评估，各负责自己那份 condition。
                if eval_manifest is not None and group_counter % eval_every == 0:
                    _run_eval(
                        config, eval_config, runtime, eval_rewards, output, eval_manifest,
                        version=trainer.policy_version, group_step=group_counter,
                        eval_logger=eval_logger, monitor=monitor,
                    )
                group_started_at = time.monotonic()
                # DataLoader worker 预取会提前推进 sampler.position，不能用它显示消费进度。
                # group_counter 只在一个 group 真正完成后递增，因此对进度与断点恢复都准确。
                completed_condition = group_counter + 1
                condition_position = (completed_condition - 1) % conditions_per_epoch + 1
                epoch_index = (completed_condition - 1) // conditions_per_epoch + 1
                current_epoch_progress = completed_condition / conditions_per_epoch
                if rank == 0:
                    progress.set_postfix(
                        epoch=f"{current_epoch_progress:.2f}/{estimated_total_epochs:.2f}",
                        condition=f"{condition_position}/{conditions_per_epoch}",
                        stage="rollout",
                    )
                version = trainer.policy_version
                runtime.set_policy_version(version)
                _assert_group_alignment(raw.condition_id, version, group_counter)
                prepared = runtime.prepare_condition(raw)
                first_seed = base_seed + group_counter * global_group_size
                global_group_id = f"{raw.condition_id}_policy_{version:08d}_seed_{first_seed}"
                all_seeds = [first_seed + offset for offset in range(global_group_size)]
                local_seeds = all_seeds[rank * local_group_size : (rank + 1) * local_group_size]
                group_dir, artifacts = runtime.rollout_group(
                    prepared, seeds=local_seeds, output_dir=output,
                    expected_group_size=local_group_size,
                    rollout_batch_size=rollout_batch_size,
                )
                if rank == 0:
                    progress.set_postfix(
                        epoch=f"{current_epoch_progress:.2f}/{estimated_total_epochs:.2f}",
                        condition=f"{condition_position}/{conditions_per_epoch}",
                        stage="reward",
                    )
                artifacts_by_seed = {artifact.seed: artifact for artifact in artifacts}
                _score_group(config, rewards, group_dir, raw, version, artifacts=artifacts_by_seed)
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
                valid_seeds = [seed for seed, _ in ordered_rewards]
                if len(valid_seeds) != len(set(valid_seeds)) or not all(seed in all_seeds for seed in valid_seeds):
                    raise RuntimeError("distributed rollout seed gather is incomplete or duplicated")
                global_rewards = [reward for _, reward in ordered_rewards]
                valid_global_reward_rows = [
                    row["reward"] for row in global_rollout_rows
                    if row["reward"].get("total_reward") is not None
                    and row["reward"].get("valid", True)
                ]
                global_action_rewards = [
                    float(reward["action_reward"]) for reward in valid_global_reward_rows
                ]
                geometry_values = [reward.get("geometry_reward") for reward in valid_global_reward_rows]
                # Action-only reward mode intentionally has no geometry reward.
                # Keep the separate-advantage diagnostics optional in that mode.
                global_geometry_rewards = (
                    [float(value) for value in geometry_values]
                    if geometry_values and all(value is not None for value in geometry_values)
                    else None
                )
                min_valid_seeds = max(2, int(config["reward"].get("min_valid_seeds_per_group", 8)))
                skip_reason = _group_skip_reason(
                    gathered_rewards, min_valid_seeds=min_valid_seeds
                )
                if skip_reason is not None:
                    # 有效（跟踪成功）seed 数低于阈值：跳过该 task，不进训练，
                    # 或任一 rank 没有本地有效样本：四卡同步跳过并推进 sampler。
                    local_valid_counts = [len(rows) for rows in gathered_rewards]
                    if rank == 0:
                        progress.write(
                            f"[WARN] skip task {raw.condition_id}: {skip_reason}; "
                            f"per-rank valid counts={local_valid_counts}"
                        )
                    if logger is not None:
                        logger.write({
                            "condition_id": raw.condition_id, "group_id": global_group_id,
                            "optimizer_step": trainer.optimizer_step, "policy_version": version,
                            "optimizer_stepped": False, "skipped": True,
                            "valid_seeds": len(valid_seeds), "group_size": global_group_size,
                            "valid_seeds_per_rank": local_valid_counts,
                            "skip_reason": skip_reason,
                        })
                    group_counter += 1
                    if world_size > 1:
                        dist.barrier()
                    retention = config.get("storage", {}).get("rollout_retention")
                    if retention is None:
                        keep_legacy = bool(config.get("storage", {}).get("keep_consumed_rollouts", True))
                        retention = "all" if keep_legacy else "none"
                    _retain_consumed_group(group_dir, output, str(retention))
                    del prepared, artifacts, artifacts_by_seed
                    if trainer.optimizer_step >= max_steps:
                        break
                    continue
                _, samples = load_fresh_rollout_group(
                    group_dir, device=runtime.device, expected_group_size=local_group_size,
                    expected_policy_version=version,
                    global_rewards=global_rewards,
                    global_action_rewards=global_action_rewards,
                    global_geometry_rewards=global_geometry_rewards,
                    action_weight=float(config["reward"].get("action_weight", 1.0)),
                    geometry_weight=float(config["reward"].get("geometry_weight", 1.0)),
                    artifacts=artifacts_by_seed,
                )
                if rank == 0:
                    progress.set_postfix(
                        epoch=f"{current_epoch_progress:.2f}/{estimated_total_epochs:.2f}",
                        condition=f"{condition_position}/{conditions_per_epoch}",
                        stage="update",
                    )
                local_metrics = trainer.update_group(
                    samples, group_id=global_group_id, normalization_count=len(valid_seeds)
                )
                metrics = _global_group_metrics(_gather_per_rank(local_metrics))
                reward_values = [
                    float(row["reward"]["total_reward"])
                    for row in global_rollout_rows
                    if row.get("reward", {}).get("total_reward") is not None
                ]
                mean_reward = (
                    float(sum(reward_values) / len(reward_values)) if reward_values else None
                )
                epoch_progress = completed_condition / conditions_per_epoch
                train_row = {
                    "condition_id": raw.condition_id,
                    "group_id": global_group_id,
                    "epoch": epoch_progress,
                    "epoch_index": epoch_index,
                    "condition_position": condition_position,
                    "conditions_per_epoch": conditions_per_epoch,
                    "estimated_total_epochs": estimated_total_epochs,
                    "group_duration_seconds": time.monotonic() - group_started_at,
                    "mean_reward": mean_reward,
                    **metrics,
                }
                if logger is not None:
                    logger.write(train_row)
                    monitor.log_group(
                        train_row, global_rollout_rows, group_step=group_counter + 1
                    )
                    if metrics["optimizer_stepped"]:
                        progress.update(1)
                    progress.set_postfix(
                        epoch=f"{epoch_progress:.2f}/{estimated_total_epochs:.2f}",
                        condition=f"{condition_position}/{conditions_per_epoch}",
                        loss=f"{float(metrics['loss']):.4g}",
                        reward="n/a" if mean_reward is None else f"{mean_reward:.4f}",
                        step_s=f"{train_row['group_duration_seconds']:.1f}",
                    )
                group_counter += 1
                if metrics["optimizer_stepped"]:
                    runtime.set_policy_version(trainer.policy_version)
                    every = int(config["checkpoint"]["every_optimizer_steps"])
                    if rank == 0 and every > 0 and trainer.optimizer_step % every == 0:
                        # 主线程只做 CPU 快照（快），序列化 + 落盘由后台线程完成，
                        # 避免 rank0 在 checkpoint I/O 上阻塞整组。
                        checkpoint_writer.submit(
                            output / "checkpoints", step=trainer.optimizer_step,
                            snapshot=build_checkpoint_snapshot(
                                step=trainer.optimizer_step,
                                policy=runtime.transformer,
                                optimizer=trainer.optimizer,
                                lr_scheduler=trainer.lr_scheduler,
                                ema_state={} if ema is None else ema.state_dict(),
                                sampler_state=_sampler_state_after_groups(sampler, group_counter),
                                trainer_state={**trainer.state_dict(), "group_counter": group_counter},
                                config=config,
                            ),
                        )
                if world_size > 1:
                    dist.barrier()
                retention = config.get("storage", {}).get("rollout_retention")
                if retention is None:
                    # 兼容旧配置：true=all，false=none。
                    keep_legacy = bool(config.get("storage", {}).get("keep_consumed_rollouts", True))
                    retention = "all" if keep_legacy else "none"
                _retain_consumed_group(group_dir, output, str(retention))
                del samples, prepared, artifacts, artifacts_by_seed
                if trainer.optimizer_step >= max_steps:
                    break
        if checkpoint_writer is not None:
            checkpoint_writer.close()


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
    parser.add_argument("--eval-prep-root")
    parser.add_argument("--eval-every-group-steps", type=int)
    parser.add_argument("--eval-max-conditions", type=int)
    parser.add_argument("--eval-seeds-per-condition", type=int)
    parser.add_argument("--eval-rollout-batch-size", type=int)
    parser.add_argument("--eval-seed", type=int)
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
        eval_manifest = _preflight_eval(config)
        if env_rank == 0:
            eval_info = (
                {"num_eval_samples": eval_manifest["num_samples"]}
                if eval_manifest is not None
                else {"eval": "disabled"}
            )
            print(json.dumps({
                "manifest_sha256": manifest["sha256"], "num_samples": manifest["num_samples"],
                **eval_info,
            }))
        return
    initialized_here = False
    train_device = args.device
    if env_world_size > 1:
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            parser.error("multi-process training requires CUDA and one visible GPU per rank")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=_distributed_timeout(),
        )
        initialized_here = True
        train_device = f"cuda:{local_rank}"
    try:
        train(config, resume=args.resume, device=train_device)
    finally:
        if initialized_here:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
