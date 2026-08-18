from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from experiments.awm_coca.advantage import local_leave_one_out_advantages
from experiments.awm_coca.gesim_adapter import GeSimConditionBatch
from experiments.awm_coca.training_core import RolloutTrainingSample


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _condition(payload: Any) -> GeSimConditionBatch:
    if isinstance(payload, GeSimConditionBatch):
        return payload
    if not isinstance(payload, dict):
        raise TypeError("condition.pt must contain GeSimConditionBatch or its field dictionary")
    return GeSimConditionBatch(**payload)


def load_fresh_rollout_group(
    group_dir: str | Path,
    *,
    device: str | torch.device,
    expected_group_size: int | None = None,
    expected_policy_version: int | None = None,
    global_rewards: list[float] | None = None,
    global_action_rewards: list[float] | None = None,
    global_geometry_rewards: list[float] | None = None,
    action_weight: float = 0.5,
    geometry_weight: float = 0.5,
    artifacts: dict[int, Any] | None = None,
) -> tuple[str, list[RolloutTrainingSample]]:
    root = Path(group_dir)
    metadata_path = root / "group.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing rollout group metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    group_id = str(metadata.get("group_id", ""))
    condition_id = str(metadata.get("condition_id", ""))
    policy_version = int(metadata.get("policy_version", -1))
    if not group_id or not condition_id or policy_version < 0:
        raise ValueError("group.json needs group_id, condition_id, and non-negative policy_version")
    if expected_policy_version is not None and policy_version != expected_policy_version:
        raise ValueError(f"stale rollout group policy_version={policy_version}, expected {expected_policy_version}")
    directories = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("seed_"))
    required_size = expected_group_size or int(metadata.get("group_size", 0))
    if required_size and len(directories) != required_size:
        raise ValueError(f"rollout group has {len(directories)} seeds, expected {required_size}")
    if len(directories) < 2:
        raise ValueError("a fresh rollout group needs at least two seed directories")
    rewards, action_rewards, geometry_rewards, payloads = [], [], [], []
    for directory in directories:
        reward = json.loads((directory / "reward.json").read_text(encoding="utf-8"))
        if reward.get("total_reward") is None or not reward.get("valid", True):
            continue  # 该 seed 奖励无效（SAM3/跟踪失败等），训练时跳过该 seed
        credit = json.loads((directory / "credit.json").read_text(encoding="utf-8"))
        seed_meta = json.loads((directory / "rollout.json").read_text(encoding="utf-8"))
        if seed_meta.get("condition_id") != condition_id or int(seed_meta.get("policy_version", -1)) != policy_version:
            raise ValueError(f"condition/policy mismatch in {directory}")
        chunks = seed_meta.get("num_chunks", 1)
        if int(chunks) != 1:
            raise ValueError(f"AWM-CoCA V2 supports exactly one rollout chunk, got {chunks}")
        total_reward = reward.get("total_reward")
        rewards.append(float(total_reward))
        action_rewards.append(float(reward["action_reward"]))
        geometry_value = reward.get("geometry_reward")
        if geometry_value is not None:
            geometry_rewards.append(float(geometry_value))
        seed = int(seed_meta["seed"])
        if artifacts is not None:
            artifact = artifacts.get(seed)
            if artifact is None:
                raise ValueError(f"missing in-memory rollout artifact for seed {seed} in {directory}")
            payloads.append((directory, credit, artifact.final_future_latent, _condition(artifact.condition_template)))
        else:
            latent_path = directory / "final_future_latent.pt"
            if not latent_path.is_file():
                raise FileNotFoundError(f"missing explicit future latent: {latent_path}")
            condition_path = directory / "condition.pt"
            if not condition_path.is_file():
                condition_path = root / "condition.pt"
            if not condition_path.is_file():
                raise FileNotFoundError(f"missing seed or shared condition: {condition_path}")
            payloads.append((directory, credit, _load_torch(latent_path), _condition(_load_torch(condition_path))))
    advantage_rewards = rewards if global_rewards is None else [float(value) for value in global_rewards]
    if len(advantage_rewards) < 2:
        raise ValueError("global leave-one-out advantage requires at least two rewards")
    advantages = local_leave_one_out_advantages(rewards, advantage_rewards)
    action_advantages: list[float | None] = [None] * len(rewards)
    geometry_advantages: list[float | None] = [None] * len(rewards)
    if global_action_rewards is not None and global_geometry_rewards is not None:
        global_action_rewards = [float(value) for value in global_action_rewards]
        global_geometry_rewards = [float(value) for value in global_geometry_rewards]
        if len(global_action_rewards) != len(advantage_rewards) or len(global_geometry_rewards) != len(advantage_rewards):
            raise ValueError("global action/geometry rewards must match global total rewards")
        action_advantages = local_leave_one_out_advantages(action_rewards, global_action_rewards)
        geometry_advantages = local_leave_one_out_advantages(geometry_rewards, global_geometry_rewards)
        weight_sum = float(action_weight) + float(geometry_weight)
        if not math.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError("action_weight + geometry_weight must be finite and positive")
        normalized_action_weight = float(action_weight) / weight_sum
        normalized_geometry_weight = float(geometry_weight) / weight_sum
        combined_advantages = [
            normalized_action_weight * action + normalized_geometry_weight * geometry
            for action, geometry in zip(action_advantages, geometry_advantages)
        ]
        if not all(math.isclose(combined, total, rel_tol=1e-6, abs_tol=1e-8)
                   for combined, total in zip(combined_advantages, advantages)):
            raise ValueError("separate action/geometry advantages do not reproduce total advantage")
        advantages = combined_advantages
    samples = []
    for reward, advantage, action_advantage, geometry_advantage, (directory, credit, latent, condition) in zip(
        rewards, advantages, action_advantages, geometry_advantages, payloads
    ):
        scores = [float(row["noise_score"]) for row in credit.get("noise_rows", [])]
        samples.append(RolloutTrainingSample(
            sample_id=directory.name, condition_id=condition_id, policy_version=policy_version,
            clean_latent=latent.detach().to(device), condition=condition, advantage=advantage,
            reward=reward, noise_scores=scores,
            action_advantage=action_advantage, geometry_advantage=geometry_advantage,
        ))
    return group_id, samples
