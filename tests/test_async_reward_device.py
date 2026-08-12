from __future__ import annotations

import threading

import torch

from experiments.awm_coca.async_reward import AsyncRewardRunner, RewardRequest


def test_reward_worker_binds_rank_local_cuda_device(monkeypatch):
    calls: list[tuple[int, int]] = []

    def set_device(index: int) -> None:
        calls.append((threading.get_ident(), index))

    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    main_thread = threading.get_ident()
    request = RewardRequest("seed_1", "condition", 0, 1, {"value": 1})

    with AsyncRewardRunner(
        lambda payload: {"total_reward": payload["value"], "valid": True},
        workers=1,
        cuda_device=3,
    ) as runner:
        result = runner.submit(request).result(timeout=5)

    assert result.reward["total_reward"] == 1
    assert calls == [(calls[0][0], 3)]
    assert calls[0][0] != main_thread
