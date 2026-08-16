from __future__ import annotations

from typing import Any


class GESimSchedulerAdapter:
    def __init__(self, scheduler: Any):
        self.scheduler = scheduler

    def flow_times(self) -> list[float]:
        sigmas = self.scheduler.sigmas.detach().cpu().double().tolist()
        return [float(s / (1.0 + s)) for s in sigmas]

    def legal_branch_timesteps(self) -> list[int]:
        times = self.flow_times()
        return [i for i, (t, nxt) in enumerate(zip(times[:-1], times[1:])) if 0.0 <= nxt < t < 1.0]

