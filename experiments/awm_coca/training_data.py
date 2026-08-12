from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

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
    rewards, payloads = [], []
    for directory in directories:
        reward = json.loads((directory / "reward.json").read_text(encoding="utf-8"))
        credit = json.loads((directory / "credit.json").read_text(encoding="utf-8"))
        seed_meta = json.loads((directory / "rollout.json").read_text(encoding="utf-8"))
        if seed_meta.get("condition_id") != condition_id or int(seed_meta.get("policy_version", -1)) != policy_version:
            raise ValueError(f"condition/policy mismatch in {directory}")
        chunks = seed_meta.get("num_chunks", 1)
        if int(chunks) != 1:
            raise ValueError(f"AWM-CoCA V2 supports exactly one rollout chunk, got {chunks}")
        total_reward = reward.get("total_reward")
        if total_reward is None:
            raise ValueError(f"invalid reward in {directory}")
        latent_path = directory / "final_future_latent.pt"
        if not latent_path.is_file():
            raise FileNotFoundError(f"missing explicit future latent: {latent_path}")
        rewards.append(float(total_reward))
        condition_path = directory / "condition.pt"
        if not condition_path.is_file():
            condition_path = root / "condition.pt"
        if not condition_path.is_file():
            raise FileNotFoundError(f"missing seed or shared condition: {condition_path}")
        payloads.append((directory, credit, _load_torch(latent_path), _condition(_load_torch(condition_path))))
    advantage_rewards = rewards if global_rewards is None else [float(value) for value in global_rewards]
    if len(advantage_rewards) < 2:
        raise ValueError("global leave-one-out advantage requires at least two rewards")
    total_reward = float(sum(advantage_rewards))
    advantages = [
        reward - (total_reward - reward) / (len(advantage_rewards) - 1)
        for reward in rewards
    ]
    samples = []
    for reward, advantage, (directory, credit, latent, condition) in zip(rewards, advantages, payloads):
        scores = [float(row["noise_score"]) for row in credit.get("noise_rows", [])]
        samples.append(RolloutTrainingSample(
            sample_id=directory.name, condition_id=condition_id, policy_version=policy_version,
            clean_latent=latent.detach().to(device), condition=condition, advantage=advantage,
            reward=reward, noise_scores=scores,
        ))
    return group_id, samples
