from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from experiments.awm_coca.gesim_adapter import GeSimConditionBatch


@dataclass(frozen=True)
class BranchGroupKey:
    condition_id: str
    prompt_id: str
    initial_seed: int
    branch_timestep: int
    video_length: int
    reward_config_sha256: str

    def group_id(self, policy_version: int) -> str:
        return (
            f"{self.condition_id}_policy_{policy_version:08d}_seed_{self.initial_seed}_"
            f"branch_{self.branch_timestep:03d}"
        )


@dataclass(frozen=True)
class OrdinaryGroupKey:
    condition_id: str
    prompt_id: str
    video_length: int
    reward_config_sha256: str
    group_sequence: int

    def group_id(self, policy_version: int) -> str:
        return (
            f"{self.condition_id}_policy_{policy_version:08d}_"
            f"ordinary_{self.group_sequence:08d}"
        )


@dataclass
class CollectedTransition:
    timestep: int
    current_latent: torch.Tensor
    next_latent: torch.Tensor
    flow_time: float
    next_flow_time: float
    eta: float
    old_log_prob: float
    old_token_log_prob: torch.Tensor
    rf_noise_std: float
    noise_weight: float
    noise_seed: int


@dataclass
class OrdinaryRollout:
    sample_id: str
    group_key: OrdinaryGroupKey
    policy_version: int
    rollout_id: int
    initial_seed: int
    seed_dir: Path
    condition: GeSimConditionBatch
    transitions: list[CollectedTransition]
    reward: dict[str, Any] | None = None
    advantage: float | None = None

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_id": self.sample_id,
            "condition_id": self.group_key.condition_id,
            "prompt_id": self.group_key.prompt_id,
            "video_length": self.group_key.video_length,
            "reward_config_sha256": self.group_key.reward_config_sha256,
            "group_sequence": self.group_key.group_sequence,
            "policy_version": self.policy_version,
            "rollout_id": self.rollout_id,
            "initial_seed": self.initial_seed,
            "transitions": [
                {
                    "timestep": action.timestep,
                    "flow_time": action.flow_time,
                    "next_flow_time": action.next_flow_time,
                    "eta": action.eta,
                    "old_log_prob": action.old_log_prob,
                    "old_token_log_prob_shape": list(action.old_token_log_prob.shape),
                    "rf_noise_std": action.rf_noise_std,
                    "noise_weight": action.noise_weight,
                    "noise_seed": action.noise_seed,
                }
                for action in self.transitions
            ],
        }
        if self.reward is not None:
            payload["reward_total"] = self.reward.get("total_reward")
            payload["reward_components"] = self.reward
        if self.advantage is not None:
            payload["advantage"] = self.advantage
        return payload


@dataclass
class BranchRollout:
    sample_id: str
    group_key: BranchGroupKey
    policy_version: int
    branch_id: int
    branch_noise_seed: int
    seed_dir: Path
    current_latent: torch.Tensor
    next_latent: torch.Tensor
    condition: GeSimConditionBatch
    flow_time: float
    next_flow_time: float
    eta: float
    old_log_prob: float
    old_token_log_prob: torch.Tensor
    rf_noise_std: float
    noise_weight: float
    prefix_latent_sha256: str
    branch_noise_sha256: str
    reward: dict[str, Any] | None = None
    advantage: float | None = None

    def metadata(self) -> dict[str, Any]:
        payload = {
            "sample_id": self.sample_id,
            "condition_id": self.group_key.condition_id,
            "prompt_id": self.group_key.prompt_id,
            "initial_seed": self.group_key.initial_seed,
            "branch_timestep": self.group_key.branch_timestep,
            "video_length": self.group_key.video_length,
            "reward_config_sha256": self.group_key.reward_config_sha256,
            "policy_version": self.policy_version,
            "branch_id": self.branch_id,
            "branch_noise_seed": self.branch_noise_seed,
            "flow_time": self.flow_time,
            "next_flow_time": self.next_flow_time,
            "eta": self.eta,
            "old_log_prob": self.old_log_prob,
            "old_token_log_prob_shape": list(self.old_token_log_prob.shape),
            "rf_noise_std": self.rf_noise_std,
            "noise_weight": self.noise_weight,
            "prefix_latent_sha256": self.prefix_latent_sha256,
            "branch_noise_sha256": self.branch_noise_sha256,
        }
        if self.reward is not None:
            payload["reward_total"] = self.reward.get("total_reward")
            payload["reward_components"] = self.reward
        if self.advantage is not None:
            payload["advantage"] = self.advantage
        return payload


@dataclass(frozen=True)
class TrainerStepRecord:
    optimizer_step: int
    policy_version: int
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
