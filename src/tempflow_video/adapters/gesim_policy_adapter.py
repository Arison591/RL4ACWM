from __future__ import annotations

from typing import Any

from tempflow_video.runtime.rng_isolation import isolated_rng


class GESimPolicyAdapter:
    """Composition-only adapter; it never patches the upstream runtime."""

    def __init__(self, runtime: Any):
        self.runtime = runtime

    def prepare_condition(self, *args, seed: int | None = None, **kwargs):
        with isolated_rng(seed):
            return self.runtime.prepare_condition(*args, **kwargs)

    def predict_velocity_or_noise(self, *args, **kwargs):
        return self.runtime.predict_velocity_or_noise(*args, **kwargs)

    def decode_video(self, *args, seed: int | None = None, **kwargs):
        with isolated_rng(seed):
            return self.runtime.decode_video(*args, **kwargs)

    def trainable_parameters(self):
        return [p for p in self.runtime.transformer.parameters() if p.requires_grad]

