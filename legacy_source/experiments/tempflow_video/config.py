from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_MODES = {
    "base_eval": (False, False, False),
    "video_grpo": (True, False, False),
    "tempflow_branch_only": (True, True, False),
    "tempflow_noise_weight_only": (True, False, True),
    "tempflow_full": (True, True, True),
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def _repo_path(value: str) -> str:
    path = Path(value)
    return str(path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve())


def load_tempflow_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        local = yaml.safe_load(handle)
    if not isinstance(local, dict):
        raise ValueError("TempFlow config must be a YAML mapping")
    base_path = local.pop("base_config", None)
    if base_path is None:
        raise ValueError("TempFlow config must declare base_config")
    base_path = Path(_expand(base_path))
    if not base_path.is_absolute():
        base_path = (REPO_ROOT / base_path).resolve()
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    config = _expand(_deep_merge(base, local))
    config["_config_path"] = str(config_path)
    config["_base_config_path"] = str(base_path)
    mode = config.get("experiment", {}).get("mode")
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"unknown experiment.mode={mode!r}; expected {sorted(EXPERIMENT_MODES)}")
    train, branching, weighting = EXPERIMENT_MODES[mode]
    tempflow = config.setdefault("tempflow", {})
    tempflow["enabled"] = train
    tempflow["trajectory_branching"] = branching
    tempflow["noise_aware_weighting"] = weighting

    path_fields = [
        (config, "output_dir"),
        (config["dataset"], "prep_root"),
        (config["model"], "gesim_config"),
        (config["model"], "checkpoint_root"),
    ]
    for mapping, key in path_fields:
        mapping[key] = _repo_path(mapping[key])
    reward = config["reward"]
    reward.setdefault("time_alignment_protocol", "legacy_frame_index_truncate")
    reward.setdefault("generated_fps", 16)
    reward.setdefault("expected_gt_fps", 30)
    reward["gt_video_template"] = _repo_path(reward["gt_video_template"])
    reward["gt_video_templates"] = {
        camera: _repo_path(template)
        for camera, template in reward.get("gt_video_templates", {}).items()
    }
    ids_file = config["dataset"].get("ids_file")
    if ids_file:
        config["dataset"]["ids_file"] = _repo_path(ids_file)
    return config


def dump_effective_config(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
