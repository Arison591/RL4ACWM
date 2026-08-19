from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist


def partition_indices(size: int, world_size: int, rank: int) -> list[int]:
    """Deterministically shard one global branch group without dropping members."""

    if size < world_size:
        raise ValueError(
            f"global branch_factor={size} must be >= distributed world_size={world_size}"
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} must lie in [0, {world_size})")
    return list(range(rank, size, world_size))


def weighted_mean_dict(
    rows: Sequence[dict[str, float]], weights: Sequence[int], *, shared_keys: Iterable[str]
) -> dict[str, float]:
    """Combine rank-local scalar metrics; gradient norms are already globally reduced."""

    if not rows or len(rows) != len(weights) or sum(weights) <= 0:
        raise ValueError("metric rows require matching positive weights")
    shared = set(shared_keys)
    output: dict[str, float] = {}
    for key in rows[0]:
        if key in shared:
            output[key] = float(rows[0][key])
        else:
            output[key] = float(
                sum(float(row[key]) * weight for row, weight in zip(rows, weights))
                / sum(weights)
            )
    return output


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    enabled: bool

    @classmethod
    def initialize(cls, expected_world_size: int) -> "DistributedContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        expected = int(expected_world_size)
        if expected < 1:
            raise ValueError("distributed.world_size must be positive")
        if world_size != expected:
            raise RuntimeError(
                f"torchrun WORLD_SIZE={world_size} does not match config world_size={expected}"
            )
        enabled = world_size > 1
        if enabled:
            if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
                raise RuntimeError(
                    f"distributed TempFlow needs {world_size} visible CUDA devices, "
                    f"found {torch.cuda.device_count()}"
                )
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return cls(rank=rank, local_rank=local_rank, world_size=world_size, enabled=enabled)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> str:
        return f"cuda:{self.local_rank}"

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier(device_ids=[self.local_rank])

    def broadcast_path(self, value: Path | None) -> Path:
        payload: list[Any] = [None if value is None else str(value)]
        if self.enabled:
            dist.broadcast_object_list(payload, src=0)
        if payload[0] is None:
            raise RuntimeError("rank 0 did not broadcast a run directory")
        return Path(payload[0])

    def gather_objects(self, value: Any) -> list[Any]:
        if not self.enabled:
            return [value]
        output: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(output, value)
        return output

    def sum_tensors_(self, tensors: Sequence[torch.Tensor]) -> None:
        if not self.enabled:
            return
        for tensor in tensors:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank])
            dist.destroy_process_group()
