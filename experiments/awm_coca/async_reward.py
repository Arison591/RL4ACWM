from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class RewardRequest:
    sample_id: str
    condition_id: str
    policy_version: int
    seed: int
    payload: Any


@dataclass(frozen=True)
class RewardResult:
    sample_id: str
    condition_id: str
    policy_version: int
    seed: int
    reward: dict[str, Any]


class AsyncRewardRunner:
    def __init__(
        self,
        reward_fn: Callable[[Any], dict[str, Any]],
        *,
        workers: int = 1,
        cuda_device: int | None = None,
    ) -> None:
        if workers <= 0:
            raise ValueError("reward workers must be positive")
        if cuda_device is not None and cuda_device < 0:
            raise ValueError("reward CUDA device must be non-negative")
        self.reward_fn = reward_fn
        self.cuda_device = cuda_device
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="awm-reward")

    def submit(self, request: RewardRequest) -> Future[RewardResult]:
        def run() -> RewardResult:
            # CUDA current device is thread-local.  torchrun binds the training
            # thread to LOCAL_RANK, but a fresh ThreadPoolExecutor worker starts
            # on logical cuda:0 unless it is bound again here.  Reward models
            # are lazy-loaded in this worker, so bind before invoking reward_fn.
            if self.cuda_device is not None:
                import torch

                torch.cuda.set_device(self.cuda_device)
            reward = self.reward_fn(request.payload)
            return RewardResult(request.sample_id, request.condition_id, request.policy_version,
                                request.seed, reward)
        return self.executor.submit(run)

    @staticmethod
    def gather(
        futures: Iterable[Future[RewardResult]],
        *,
        condition_id: str,
        policy_version: int,
        timeout: float | None = None,
    ) -> list[RewardResult]:
        results = [future.result(timeout=timeout) for future in futures]
        if not results:
            raise ValueError("reward group is empty")
        sample_ids = set()
        for result in results:
            if result.condition_id != condition_id or result.policy_version != policy_version:
                raise ValueError("reward result condition/policy version mismatch")
            if result.sample_id in sample_ids:
                raise ValueError(f"duplicate reward sample id: {result.sample_id}")
            sample_ids.add(result.sample_id)
            # 单个 seed 的奖励可能无效（如 SAM3 在生成帧上分割不出目标、跟踪退化）；
            # 不在此抛错 —— 由调用方把无效 seed 标记并跳过该 seed / 整个 task。
        return sorted(results, key=lambda item: item.seed)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "AsyncRewardRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
