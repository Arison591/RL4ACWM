from __future__ import annotations

import json
import os
import platform
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import torch

from experiments.awm_coca.condition_dataset import CAMERAS, build_manifest
from experiments.tempflow_video.dynamics import noise_aware_weights, raw_noise_levels


_UNEXPANDED = re.compile(r"\$\{[^}]+\}")


def _require_file(path: Path, label: str, checks: list[dict[str, Any]]) -> None:
    exists = path.is_file()
    checks.append({"name": label, "ok": exists, "path": str(path)})
    if not exists:
        raise FileNotFoundError(f"{label} is missing: {path}")


def _require_dir(path: Path, label: str, checks: list[dict[str, Any]]) -> None:
    exists = path.is_dir()
    checks.append({"name": label, "ok": exists, "path": str(path)})
    if not exists:
        raise FileNotFoundError(f"{label} is missing: {path}")


def _find_unexpanded(value: Any, prefix: str = "config") -> list[str]:
    if isinstance(value, str):
        return [prefix] if _UNEXPANDED.search(value) else []
    if isinstance(value, dict):
        return sum((_find_unexpanded(item, f"{prefix}.{key}") for key, item in value.items()), [])
    if isinstance(value, list):
        return sum((_find_unexpanded(item, f"{prefix}[{index}]") for index, item in enumerate(value)), [])
    return []


def _validate_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open GT video: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frames < 29 or width <= 0 or height <= 0:
        raise ValueError(f"invalid GT video metadata: {path} frames={frames} size={width}x{height}")
    return {"path": str(path), "frames": frames, "width": width, "height": height, "fps": fps}


def _scheduler_report(config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from gesim_video_gen_examples.infer_gesim import load_config
    from utils import import_custom_class

    model = config["model"]
    args = load_config(model["gesim_config"])
    root = Path(model["checkpoint_root"])
    scheduler_class = import_custom_class(
        args.diffusion_scheduler_class,
        str((Path(config["_base_config_path"]).parents[1] / args.diffusion_scheduler_class_path).resolve()),
    )
    scheduler_dir = root / Path(args.pretrained_model_name_or_path).name / "scheduler"
    scheduler = scheduler_class.from_pretrained(str(scheduler_dir))
    scheduler.register_to_config(
        sigma_max=80.0, sigma_min=0.002, sigma_data=1.0, final_sigmas_type="sigma_min"
    )
    steps = int(config["rollout"]["reverse_denoise_steps"])
    scheduler.set_timesteps(sigmas=np.linspace(0, 1, steps), device="cpu")
    if scheduler.config.final_sigmas_type == "sigma_min":
        scheduler.sigmas[-1] = scheduler.sigmas[-2]
    sigmas = torch.as_tensor(scheduler.sigmas, dtype=torch.float64)
    times = sigmas / (sigmas + 1.0)
    eta = float(config["tempflow"]["eta"])
    raw = raw_noise_levels(times.tolist(), eta=eta)
    weights = noise_aware_weights(
        times.tolist(),
        eta=eta,
        enabled=bool(config["tempflow"]["noise_aware_weighting"]),
        normalization=config["tempflow"].get("noise_weight_normalization", "schedule_mean"),
    )
    branchable = [index for index, value in enumerate(raw.tolist()) if value > 0]
    configured_values = config["tempflow"].get("branch_timesteps")
    if configured_values is None:
        fraction = float(config["tempflow"].get("timestep_fraction", 0.99))
        if not 0.0 < fraction <= 1.0:
            raise ValueError("tempflow.timestep_fraction must lie in (0, 1]")
        configured = [
            index
            for index in range(int(steps * fraction))
            if index in branchable
        ]
    else:
        configured = [int(value) for value in configured_values]
    invalid = sorted(set(configured).difference(branchable))
    if invalid:
        raise ValueError(f"configured branch timesteps are not valid reverse transitions: {invalid}")
    if not configured or len(set(configured)) != len(configured):
        raise ValueError("configured branch timesteps must be non-empty and unique")
    return {
        "reverse_denoise_steps": steps,
        "edm_sigmas": sigmas.tolist(),
        "flow_times": times.tolist(),
        "raw_transition_noise": raw.tolist(),
        "noise_weights": weights.tolist(),
        "branchable_timesteps": branchable,
        "selected_branch_timesteps": configured,
        "terminal_duplicate_noop": bool(times[-1] == times[-2]),
    }


def run_preflight(config: dict[str, Any], *, load_model: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    unresolved = _find_unexpanded(config)
    if unresolved:
        raise ValueError(f"unexpanded environment variables in: {unresolved}")
    branch_factor = int(config["tempflow"].get("branch_factor", 0))
    if config["tempflow"]["enabled"] and branch_factor < 2:
        raise ValueError(f"GRPO branch/group factor must be at least 2, got {branch_factor}")

    dataset_cfg = config["dataset"]
    ids_path = Path(dataset_cfg["ids_file"])
    _require_file(ids_path, "condition id list", checks)
    expected_ids = [
        line.strip()
        for line in ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(expected_ids) != 16 or len(set(expected_ids)) != 16:
        raise ValueError(f"overfit id list must contain exactly 16 unique ids, got {len(expected_ids)}")
    manifest, invalid = build_manifest(
        dataset_cfg["prep_root"],
        include_samples=dataset_cfg.get("include_samples", ()),
        exclude_samples=dataset_cfg.get("exclude_samples", ()),
        limit=0,
        validation_mode=dataset_cfg.get("validation_mode", "strict"),
    )
    actual_ids = [row["condition_id"] for row in manifest["samples"]]
    if set(actual_ids) != set(expected_ids):
        raise ValueError(
            f"prep set does not exactly match ids file; missing={sorted(set(expected_ids)-set(actual_ids))} "
            f"extra={sorted(set(actual_ids)-set(expected_ids))}"
        )
    if invalid:
        raise ValueError(f"invalid prep samples: {invalid}")

    gt_videos = []
    templates = config["reward"]["gt_video_templates"]
    for condition_id in expected_ids:
        for camera in CAMERAS:
            path = Path(templates[camera].format(condition_id=condition_id, camera=camera))
            _require_file(path, f"GT {condition_id}/{camera}", checks)
            gt_videos.append(_validate_video(path))

    checkpoint_root = Path(config["model"]["checkpoint_root"])
    asset_root = checkpoint_root.parent
    os.environ["AWM_ASSET_ROOT"] = str(asset_root)
    _require_file(checkpoint_root / "gesim/ge_sim_cosmos_v0.1.safetensors", "GE-Sim checkpoint", checks)
    cosmos = checkpoint_root / "Cosmos-Predict2-2B-Video2World"
    _require_file(cosmos / "text_encoder/model.safetensors.index.json", "Cosmos text encoder", checks)
    _require_file(cosmos / "vae/diffusion_pytorch_model.safetensors", "Cosmos VAE", checks)
    _require_file(cosmos / "scheduler/scheduler_config.json", "Cosmos scheduler", checks)
    _require_file(checkpoint_root / "sam3.pt", "SAM3 checkpoint", checks)
    _require_file(checkpoint_root / "yoloworld-EWMBench-v0.1.pt", "YOLO-World checkpoint", checks)
    _require_file(checkpoint_root / "cowtracker/cowtracker_model.pth", "CoWTracker checkpoint", checks)
    _require_dir(asset_root / "third_party/sam3", "SAM3 source", checks)
    _require_dir(asset_root / "third_party/cowtracker", "CoWTracker source", checks)

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".tempflow_write_test_", dir=output, delete=True):
        pass
    checks.append({"name": "output writable", "ok": True, "path": str(output)})

    cuda = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        cuda.update(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    if not cuda["available"]:
        raise RuntimeError("TempFlow video execution requires CUDA")
    expected_world_size = int(config.get("distributed", {}).get("world_size", 1))
    if expected_world_size > 1 and cuda["device_count"] != expected_world_size:
        raise RuntimeError(
            f"distributed config requires {expected_world_size} visible GPUs, "
            f"found {cuda['device_count']}"
        )

    report: dict[str, Any] = {
        "ok": True,
        "platform": platform.platform(),
        "configuration": {
            "config_path": config["_config_path"],
            "base_config_path": config["_base_config_path"],
            "experiment_mode": config["experiment"]["mode"],
            "tempflow_enabled": config["tempflow"]["enabled"],
            "trajectory_branching": config["tempflow"]["trajectory_branching"],
            "noise_aware_weighting": config["tempflow"]["noise_aware_weighting"],
            "branch_factor": int(config["tempflow"]["branch_factor"]),
        },
        "checks": checks,
        "cuda": cuda,
        "dataset": {
            "num_samples": manifest["num_samples"],
            "manifest_sha256": manifest["sha256"],
            "condition_ids": expected_ids,
            "camera_order": manifest["camera_order"],
            "gt_video_count": len(gt_videos),
            "gt_unique_metadata": sorted(
                {f'{row["width"]}x{row["height"]}:{row["frames"]}@{row["fps"]:.3f}' for row in gt_videos}
            ),
        },
        "reward": {
            "mode": config["reward"].get("mode"),
            "time_alignment_protocol": config["reward"]["time_alignment_protocol"],
            "generated_fps": config["reward"]["generated_fps"],
            "expected_gt_fps": config["reward"]["expected_gt_fps"],
            "action_metric_weights": config["reward"].get("action_metric_weights"),
            "action_weight": config["reward"].get("action_weight"),
            "geometry_weight": config["reward"].get("geometry_weight"),
        },
        "scheduler": _scheduler_report(config),
        "model_loaded": False,
    }
    if load_model:
        from experiments.awm_coca.gesim_runtime import PersistentGeSimRuntime
        from experiments.tempflow_video.policy import ReferencePolicyAdapter, VideoPolicyAdapter

        torch.cuda.reset_peak_memory_stats()
        runtime = PersistentGeSimRuntime(config, device="cuda")
        policy = VideoPolicyAdapter(runtime)
        reference = ReferencePolicyAdapter(policy)
        reference.assert_unchanged()
        trainable = policy.get_trainable_parameters()
        report["model_loaded"] = True
        report["model"] = {
            "class": type(runtime.transformer).__name__,
            "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
            "total_parameter_count": sum(parameter.numel() for parameter in runtime.transformer.parameters()),
            "reference_parameter_count": sum(
                parameter.numel() for _, parameter in policy.reference_parameters()
            ),
            "lora_target_count": len(runtime.lora_targets),
            "lora_targets": list(runtime.lora_targets),
            "dtype": str(next(runtime.transformer.parameters()).dtype),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "reference_frozen": True,
        }
    return report


def write_preflight_report(report: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
