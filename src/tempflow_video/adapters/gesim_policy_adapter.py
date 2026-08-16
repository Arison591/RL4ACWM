from __future__ import annotations

from typing import Any

from tempflow_video.runtime.rng_isolation import isolated_rng
from tempflow_video.runtime.upstream_loader import import_upstream


class GESimPolicyAdapter:
    """Composition-only adapter; it never patches the upstream runtime."""

    def __init__(self, runtime: Any):
        self.runtime = runtime
        velocity_cls = import_upstream("experiments.awm_coca.gesim_adapter").GeSimVelocityAdapter
        self.velocity_adapter = velocity_cls(runtime.transformer)

    def prepare_condition(self, *args, seed: int | None = None, **kwargs):
        with isolated_rng(seed):
            return self.runtime.prepare_condition(*args, **kwargs)

    def predict_velocity_or_noise(self, *args, **kwargs):
        import torch
        if len(args) >= 2 and not torch.is_tensor(args[1]):
            args = (args[0], torch.tensor([args[1]], device=args[0].device, dtype=args[0].dtype), *args[2:])
        reference = bool(kwargs.pop("reference", False))
        method = self.velocity_adapter.reference_velocity if reference else self.velocity_adapter.policy_velocity
        return method(*args, **kwargs)

    def decode_video(self, *args, seed: int | None = None, **kwargs):
        with isolated_rng(seed):
            latent = args[0]
            vae, scheduler = self.runtime.vae, self.runtime.scheduler
            import torch
            mean = torch.as_tensor(vae.config.latents_mean, device=latent.device, dtype=latent.dtype).view(1, vae.config.z_dim, 1, 1, 1)
            std = torch.as_tensor(vae.config.latents_std, device=latent.device, dtype=latent.dtype).view(1, vae.config.z_dim, 1, 1, 1)
            return vae.decode((latent * std / scheduler.config.sigma_data + mean).to(vae.dtype), return_dict=False)[0]

    def trainable_parameters(self):
        return [p for p in self.runtime.transformer.parameters() if p.requires_grad]
