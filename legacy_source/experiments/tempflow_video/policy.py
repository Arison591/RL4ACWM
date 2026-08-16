from __future__ import annotations

import hashlib
from typing import Iterable

import torch

from experiments.awm_coca.gesim_adapter import GeSimConditionBatch, GeSimVelocityAdapter
from experiments.awm_coca.gesim_runtime import PersistentGeSimRuntime
from experiments.tempflow_video.dynamics import (
    FlowTransition,
    deterministic_edm_step,
    edm_sde_transition_with_logprob,
)


class VideoPolicyAdapter:
    """Adapter between GE-Sim's EDM latent convention and TempFlow RF dynamics."""

    def __init__(self, runtime: PersistentGeSimRuntime) -> None:
        self.runtime = runtime
        self.policy_model = runtime.transformer
        self.velocity_adapter = GeSimVelocityAdapter(runtime.transformer)

    def predict_velocity_or_noise(
        self,
        latent: torch.Tensor,
        flow_time: float | torch.Tensor,
        condition: GeSimConditionBatch,
        *,
        reference: bool = False,
    ) -> torch.Tensor:
        if not torch.is_tensor(flow_time):
            flow_time = torch.tensor([flow_time], device=latent.device, dtype=latent.dtype)
        if reference:
            return self.velocity_adapter.reference_velocity(latent, flow_time, condition)
        return self.velocity_adapter.policy_velocity(latent, flow_time, condition)

    def sample_one_step(
        self,
        latent: torch.Tensor,
        condition: GeSimConditionBatch,
        *,
        flow_time: float,
        next_flow_time: float,
        stochastic: bool,
        eta: float,
        generator: torch.Generator | None = None,
        next_sample: torch.Tensor | None = None,
        reference: bool = False,
    ) -> torch.Tensor | FlowTransition:
        velocity = self.predict_velocity_or_noise(
            latent, flow_time, condition, reference=reference
        )
        if stochastic:
            return edm_sde_transition_with_logprob(
                latent,
                velocity,
                flow_time=flow_time,
                next_flow_time=next_flow_time,
                eta=eta,
                generator=generator,
                next_sample=next_sample,
            )
        if next_sample is not None:
            raise ValueError("next_sample is only valid for a stochastic transition")
        return deterministic_edm_step(
            latent,
            velocity,
            flow_time=flow_time,
            next_flow_time=next_flow_time,
        )

    @torch.no_grad()
    def decode_video(self, final_edm_latent: torch.Tensor) -> torch.Tensor:
        vae = self.runtime.vae
        scheduler = self.runtime.scheduler
        latents_mean = torch.as_tensor(
            vae.config.latents_mean,
            device=final_edm_latent.device,
            dtype=final_edm_latent.dtype,
        ).view(1, vae.config.z_dim, 1, 1, 1)
        latents_std = torch.as_tensor(
            vae.config.latents_std,
            device=final_edm_latent.device,
            dtype=final_edm_latent.dtype,
        ).view(1, vae.config.z_dim, 1, 1, 1)
        vae_latent = final_edm_latent * latents_std / scheduler.config.sigma_data + latents_mean
        return vae.decode(vae_latent.to(vae.dtype), return_dict=False)[0]

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        parameters = [parameter for parameter in self.policy_model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("video policy has no trainable parameters")
        return parameters

    def get_reference_prediction(
        self,
        latent: torch.Tensor,
        flow_time: float | torch.Tensor,
        condition: GeSimConditionBatch,
    ) -> torch.Tensor:
        return self.predict_velocity_or_noise(latent, flow_time, condition, reference=True)

    def reference_parameters(self) -> Iterable[tuple[str, torch.nn.Parameter]]:
        """The frozen base parameters used when PEFT adapters are disabled."""

        for name, parameter in self.policy_model.named_parameters():
            if "lora_" not in name:
                yield name, parameter

    def reference_digest(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in self.reference_parameters():
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


class ReferencePolicyAdapter:
    """Read-only view of the initial base policy with LoRA disabled."""

    def __init__(self, policy: VideoPolicyAdapter) -> None:
        self.policy = policy
        trainable_reference = [name for name, parameter in policy.reference_parameters() if parameter.requires_grad]
        if trainable_reference:
            raise ValueError(f"reference base parameters must be frozen; first={trainable_reference[0]}")
        self._initial_versions = {
            name: int(parameter._version) for name, parameter in policy.reference_parameters()
        }

    def predict_velocity_or_noise(
        self,
        latent: torch.Tensor,
        flow_time: float | torch.Tensor,
        condition: GeSimConditionBatch,
    ) -> torch.Tensor:
        return self.policy.get_reference_prediction(latent, flow_time, condition)

    def assert_unchanged(self) -> None:
        for name, parameter in self.policy.reference_parameters():
            if int(parameter._version) != self._initial_versions[name]:
                raise RuntimeError(f"reference policy parameter changed: {name}")
