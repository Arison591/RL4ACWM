from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiments.awm_coca.run_train import _gather_per_rank
from experiments.awm_coca.trainer import _synchronize_gradients


def _four_rank_worker(rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        parameter = torch.nn.Parameter(torch.tensor([0.0]))
        # Simulate one rank not using a trainable parameter.  All four ranks
        # must still execute the same collective and receive the same average.
        if rank < world_size - 1:
            parameter.grad = torch.tensor([float(rank + 1)])
        _synchronize_gradients([parameter])
        gathered = _gather_per_rank({"rank": rank})
        Path(output_dir, f"rank_{rank}.json").write_text(
            json.dumps({
                "gradient": float(parameter.grad.item()),
                "gathered": gathered,
            }),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def test_four_rank_gradient_and_metadata_collectives(tmp_path):
    world_size = 4
    init_file = tmp_path / "gloo_init"
    mp.spawn(
        _four_rank_worker,
        args=(world_size, str(init_file), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    rows = [
        json.loads((tmp_path / f"rank_{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    # Mean of rank gradients [1, 2, 3, 0].
    assert [row["gradient"] for row in rows] == [1.5] * world_size
    assert all(
        [item["rank"] for item in row["gathered"]] == list(range(world_size))
        for row in rows
    )
