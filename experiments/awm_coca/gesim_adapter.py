from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from experiments.awm_coca.video_losses import masked_video_mse
from utils.geometry_utils import resize_traj_and_ray


@dataclass
class GeSimConditionBatch:
    memory_latents: torch.Tensor
    prompt_embeds: torch.Tensor
    cond_to_concat: torch.Tensor
    condition_indicator: torch.Tensor
    condition_mask: torch.Tensor
    padding_mask: torch.Tensor
    fps: int
    n_view: int
    n_previous: int
    num_future_latent_frames: int
    sigma_conditioning: float = 0.0001
    valid_future_mask: torch.Tensor | None = None

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "GeSimConditionBatch":
        def move(tensor: torch.Tensor, *, force_float: bool = False) -> torch.Tensor:
            return tensor.to(device=device, dtype=torch.float32 if force_float else dtype)
        return GeSimConditionBatch(
            memory_latents=move(self.memory_latents),
            prompt_embeds=move(self.prompt_embeds),
            cond_to_concat=move(self.cond_to_concat),
            condition_indicator=move(self.condition_indicator),
            condition_mask=move(self.condition_mask),
            padding_mask=move(self.padding_mask),
            fps=self.fps,
            n_view=self.n_view,
            n_previous=self.n_previous,
            num_future_latent_frames=self.num_future_latent_frames,
            sigma_conditioning=self.sigma_conditioning,
            valid_future_mask=None if self.valid_future_mask is None else move(self.valid_future_mask, force_float=True),
        )


def _video_output(value: Any) -> torch.Tensor:
    output = value[0] if isinstance(value, tuple) else value
    if isinstance(output, dict):
        output = output["video"]
    elif hasattr(output, "video"):
        output = output.video
    if not torch.is_tensor(output):
        raise TypeError(f"unsupported GE-Sim transformer output: {type(output)!r}")
    return output


class GeSimVelocityAdapter:
    def __init__(self, policy_model: torch.nn.Module) -> None:
        self.policy_model = policy_model

    def _velocity(
        self,
        noisy_future: torch.Tensor,
        flow_time: torch.Tensor,
        condition: GeSimConditionBatch,
        *,
        reference: bool,
    ) -> torch.Tensor:
        model = self.policy_model
        dtype = next(model.parameters()).dtype
        device = noisy_future.device
        condition = condition.to(device, dtype)
        time = flow_time.float().reshape(-1)
        if time.numel() == 1:
            time = time.expand(noisy_future.shape[0])
        if time.numel() != noisy_future.shape[0]:
            raise ValueError("flow_time batch does not match noisy latent batch")
        shaped_time = time.to(noisy_future.dtype).reshape(-1, 1, 1, 1, 1)
        c_in = 1.0 - shaped_time
        scaled_future = noisy_future * c_in
        model_latent = torch.cat((condition.memory_latents, scaled_future), dim=2)
        resized = resize_traj_and_ray(
            condition.cond_to_concat,
            mem_size=condition.memory_latents.shape[2],
            future_size=noisy_future.shape[2],
            height=model_latent.shape[-2],
            width=model_latent.shape[-1],
        )
        model_input = torch.cat((model_latent, resized.to(model_latent)), dim=1).to(dtype)
        full_time = shaped_time.expand(-1, 1, model_input.shape[2], 1, 1)
        conditioning_time = condition.sigma_conditioning / (condition.sigma_conditioning + 1.0)
        model_time = condition.condition_indicator * conditioning_time + (1.0 - condition.condition_indicator) * full_time
        context = model.disable_adapter() if reference and hasattr(model, "disable_adapter") else nullcontext()
        with context:
            output = model(
                hidden_states=model_input,
                timestep=model_time.to(dtype),
                encoder_hidden_states=condition.prompt_embeds,
                fps=condition.fps,
                condition_mask=condition.condition_mask,
                padding_mask=condition.padding_mask,
                return_dict=False,
                height=model_input.shape[-2],
                width=model_input.shape[-1],
                n_view=condition.n_view,
                num_frames=model_input.shape[2],
                return_video=True,
            )
        raw_future = _video_output(output)[:, :, condition.memory_latents.shape[2] :]
        # 直接返回 transformer 原生 flow-matching 输出（target = noise - clean），
        # 不做 sampler 侧 (noisy - predicted_clean)/sigma 的速度转换。
        return raw_future.float()

    def policy_velocity(self, noisy_latent: torch.Tensor, noise_time: torch.Tensor, condition: GeSimConditionBatch) -> torch.Tensor:
        return self._velocity(noisy_latent, noise_time, condition, reference=False)

    @torch.no_grad()
    def reference_velocity(self, noisy_latent: torch.Tensor, noise_time: torch.Tensor, condition: GeSimConditionBatch) -> torch.Tensor:
        return self._velocity(noisy_latent, noise_time, condition, reference=True)

    @staticmethod
    def velocity_mse(prediction: torch.Tensor, target: torch.Tensor, condition: GeSimConditionBatch) -> torch.Tensor:
        return masked_video_mse(prediction, target, n_view=condition.n_view,
                                valid_future_mask=condition.valid_future_mask)
