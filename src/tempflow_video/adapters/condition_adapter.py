from __future__ import annotations

from typing import Any

from tempflow_video.runtime.rng_isolation import isolated_rng


class ConditionAdapter:
    def __init__(self, runtime: Any):
        self.runtime = runtime

    def prepare(self, *args, seed: int | None = None, **kwargs):
        with isolated_rng(seed):
            return self.runtime.prepare_condition(*args, **kwargs)

