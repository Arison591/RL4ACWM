from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class BranchGroupKey:
    condition_id: str
    initial_seed: int
    branch_timestep: int
    reward_config_sha256: str

    def group_id(self, policy_version: int) -> str:
        raw = f"{self.condition_id}|{self.initial_seed}|{self.branch_timestep}|{self.reward_config_sha256}|{policy_version}"
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class BranchRecord:
    group_key: BranchGroupKey
    branch_id: int
    branch_noise_seed: int
    policy_version: int
    prefix_hash: str
    noise_hash: str
    next_latent: torch.Tensor


def validate_branch_group(records: list[BranchRecord]) -> None:
    if len(records) < 2:
        raise ValueError("branch group requires at least two records")
    versions = {r.policy_version for r in records}
    ids = {r.group_key.group_id(r.policy_version) for r in records}
    if len(versions) != 1 or len(ids) != 1:
        raise ValueError("group key or policy version mismatch")
    if len({r.branch_id for r in records}) != len(records):
        raise ValueError("duplicate branch id")
    if len({r.branch_noise_seed for r in records}) != len(records):
        raise ValueError("duplicate branch noise seed")
    if len({r.prefix_hash for r in records}) != 1:
        raise ValueError("deterministic prefix differs within group")
    if len({r.noise_hash for r in records}) != len(records):
        raise ValueError("branch noise must differ")

