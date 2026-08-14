from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class ResumableConditionSampler(Sampler[int]):
    def __init__(self, size: int, *, seed: int = 0, rank: int = 0, world_size: int = 1, shuffle: bool = True) -> None:
        if size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid sampler size/rank/world_size")
        self.size = size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.epoch = 0
        self.position = 0

    def _indices(self) -> list[int]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.size, generator=generator).tolist()
        else:
            indices = list(range(self.size))
        return indices[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[int]:
        indices = self._indices()
        while self.position < len(indices):
            index = indices[self.position]
            self.position += 1
            yield index
        self.epoch += 1
        self.position = 0

    def __len__(self) -> int:
        return len(range(self.rank, self.size, self.world_size))

    def state_dict(self) -> dict[str, int | bool]:
        return {"epoch": self.epoch, "position": self.position, "seed": self.seed, "shuffle": self.shuffle,
                "size": self.size, "rank": self.rank, "world_size": self.world_size}

    def load_state_dict(self, state: dict[str, int | bool]) -> None:
        for key in ("size", "rank", "world_size"):
            if int(state[key]) != getattr(self, key):
                raise ValueError(f"sampler {key} mismatch")
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
