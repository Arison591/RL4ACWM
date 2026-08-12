from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from einops import rearrange

from experiments.awm_coca.condition_dataset import RawConditionSample
from experiments.awm_coca.gesim_adapter import GeSimConditionBatch
from experiments.awm_coca.lora_utils import install_lora
from gesim_video_gen_examples.infer_gesim import load_cam_infos, load_config, load_images, prepare_model
from utils import save_video
from utils.get_ray_maps import get_ray_maps
from utils.get_traj_maps import get_traj_maps, simple_radius_gen_func


DEFAULT_PROMPT = "best quality, consistent and smooth motion, realistic, clear and distinct."


@dataclass
class PreparedGeSimCondition:
    condition_id: str
    sample_dir: str
    observation: torch.Tensor
    cond_to_concat: torch.Tensor
    original_trajectory: torch.Tensor
    memory_latents: torch.Tensor | None = None
    prompt_embeds: torch.Tensor | None = None
    condition_template: GeSimConditionBatch | None = None


class PersistentGeSimRuntime:
    """GE-Sim process runtime whose transformer and LoRA survive optimizer steps."""

    def __init__(self, config: dict[str, Any], *, device: str = "cuda") -> None:
        self.config = config
        self.device = torch.device(device)
        model_config = config["model"]
        self.args = load_config(model_config["gesim_config"])
        repo_root = Path(__file__).resolve().parents[2]

        def repo_path(value: str) -> str:
            expanded = Path(os.path.expandvars(value)).expanduser()
            return str(expanded.resolve() if expanded.is_absolute() else (repo_root / expanded).resolve())

        checkpoint_root = repo_path(model_config.get("checkpoint_root", "checkpoints"))

        def checkpoint_path(value: str) -> str:
            expanded = Path(os.path.expandvars(value)).expanduser()
            if expanded.is_absolute():
                return str(expanded.resolve())
            parts = expanded.parts
            relative = Path(*parts[1:]) if parts and parts[0] == "checkpoints" else expanded
            return str((Path(checkpoint_root) / relative).resolve())

        for name in ("pretrained_model_name_or_path", "tokenizer_pretrained_model_name_or_path", "vae_path"):
            if getattr(self.args, name, None):
                setattr(self.args, name, checkpoint_path(getattr(self.args, name)))
        for name in ("vae_class_path", "diffusion_model_class_path", "diffusion_scheduler_class_path", "pipeline_class_path"):
            value = getattr(self.args, name, None)
            if value and str(value).endswith(".py"):
                setattr(self.args, name, repo_path(value))
        self.args.diffusion_model["model_path"] = checkpoint_path(self.args.diffusion_model["model_path"])
        dtype = torch.bfloat16 if model_config.get("dtype", "bf16") == "bf16" else torch.float32
        _, self.text_encoder, self.vae, transformer, self.scheduler, self.pipe = prepare_model(
            self.args, dtype=dtype, device=device
        )
        self.transformer, self.lora_targets = install_lora(
            transformer,
            rank=int(model_config["lora_rank"]), alpha=int(model_config["lora_alpha"]),
            dropout=float(model_config.get("lora_dropout", 0.0)), init=model_config.get("lora_init", "gaussian"),
            target_modules=model_config.get("lora_targets"),
        )
        if bool(model_config.get("gradient_checkpointing", True)):
            base_transformer = (
                self.transformer.get_base_model()
                if hasattr(self.transformer, "get_base_model")
                else self.transformer
            )
            base_transformer.enable_gradient_checkpointing()
        self.pipe.transformer = self.transformer
        self.transformer.train()
        self.policy_version = 0
        rollout = config["rollout"]
        if int(rollout["chunks"]) != 1 or int(rollout["history_frames"]) != 4 or int(rollout["future_frames"]) != 25:
            raise ValueError("PersistentGeSimRuntime V2 requires exactly 4 history + 25 future frames")
        if int(self.args.data["train"]["chunk"]) != 25 or int(self.args.num_inference_step) != int(rollout["reverse_denoise_steps"]):
            raise ValueError("GE-Sim config disagrees with rollout chunk/denoise settings")

    def prepare_condition(self, raw: RawConditionSample) -> PreparedGeSimCondition:
        valid_cams = [f"{name}_color" for name in self.args.data["train"]["valid_cam"]]
        sample_size = self.args.data["train"]["sample_size"]
        observation, original_sizes = load_images(
            self.args, raw.sample_dir, valid_cams, size=(sample_size[1], sample_size[0])
        )
        extrinsics, intrinsics = load_cam_infos(
            raw.sample_dir, raw.sample_dir, self.args.data["train"]["valid_cam"],
            orisize=original_sizes, size=sample_size,
        )
        extrinsics = torch.as_tensor(extrinsics, dtype=torch.float32)
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32)
        actions = raw.actions.float()
        trajectory = get_traj_maps(
            actions, torch.linalg.inv(extrinsics), extrinsics, intrinsics, sample_size,
            radius_gen_func=simple_radius_gen_func,
        ) * 2 - 1
        rays_o, rays_d = get_ray_maps(
            intrinsics.unsqueeze(1).repeat(1, extrinsics.shape[1], 1, 1).reshape(-1, 3, 3),
            extrinsics.reshape(-1, 4, 4), sample_size[0], sample_size[1],
        )
        rays = torch.cat((rays_o, rays_d), dim=-1).reshape(
            trajectory.shape[1], trajectory.shape[2], rays_o.shape[1], rays_o.shape[2], -1
        ).permute(4, 0, 1, 2, 3)
        full_condition = torch.cat((trajectory, rays), dim=0)
        history = int(self.config["rollout"]["history_frames"])
        future = int(self.config["rollout"]["future_frames"])
        condition = torch.cat((full_condition[:, :, :history], full_condition[:, :, history:history + future]), dim=2)
        return PreparedGeSimCondition(raw.condition_id, raw.sample_dir, observation, condition, trajectory)

    def rollout_group(
        self,
        prepared: PreparedGeSimCondition,
        *,
        seeds: Sequence[int],
        output_dir: str | Path,
        prompt: str = DEFAULT_PROMPT,
        expected_group_size: int | None = None,
    ) -> Path:
        required_size = (
            int(self.config["rollout"]["group_size"])
            if expected_group_size is None
            else int(expected_group_size)
        )
        if len(seeds) != required_size or len(set(seeds)) != len(seeds):
            raise ValueError("rollout seeds must be unique and match group_size")
        group_id = f"{prepared.condition_id}_policy_{self.policy_version:08d}_seed_{int(seeds[0])}"
        group_dir = Path(output_dir) / "rollouts" / group_id
        group_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "group_id": group_id, "condition_id": prepared.condition_id,
            "policy_version": self.policy_version, "group_size": len(seeds), "seeds": list(seeds),
        }
        (group_dir / "group.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.transformer.eval()
        try:
            for seed in seeds:
                self._rollout_seed(prepared, seed=seed, seed_dir=group_dir / f"seed_{seed}", prompt=prompt)
        finally:
            self.transformer.train()
        return group_dir

    @torch.inference_mode()
    def _rollout_seed(self, prepared: PreparedGeSimCondition, *, seed: int, seed_dir: Path, prompt: str) -> None:
        seed_dir.mkdir(parents=True, exist_ok=False)
        observation = prepared.observation
        views, _, _, height, width = observation.shape
        trajectory: list[dict[str, Any]] = []
        captured_condition: dict[str, torch.Tensor] = {}

        def on_start(_pipe: Any, index: int, timestep: torch.Tensor, values: dict[str, torch.Tensor]) -> None:
            if index == 0:
                trajectory.append({"step": 0, "timestep": float(timestep.item()), "latents": values["latents"].detach().cpu()})
                for key in ("conditioning_latents", "cond_indicator", "cond_mask", "padding_mask", "cond_to_concat", "prompt_embeds"):
                    captured_condition[key] = values[key].detach().cpu()

        def on_end(_pipe: Any, index: int, timestep: torch.Tensor, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            trajectory.append({"step": index + 1, "timestep": float(timestep.item()), "latents": values["latents"].detach().cpu()})
            return values

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        result = self.pipe.infer(
            video=observation.permute(0, 2, 1, 3, 4).to(self.device),
            cond_to_concat=rearrange(prepared.cond_to_concat, "c v t h w -> v c t h w"),
            prompt=None if prepared.prompt_embeds is not None else [prompt],
            prompt_embeds=None if prepared.prompt_embeds is None else prepared.prompt_embeds.to(self.device),
            negative_prompt=None, height=height, width=width, n_view=views,
            num_frames=int(self.config["rollout"]["future_frames"]),
            num_inference_steps=int(self.config["rollout"]["reverse_denoise_steps"]),
            n_prev=int(self.config["rollout"]["history_frames"]), guidance_scale=1.0,
            generator=generator,
            conditioning_latents=None if prepared.memory_latents is None else prepared.memory_latents.to(self.device),
            callback_on_step_start=on_start, callback_on_step_end=on_end,
            output_type="pt", postprocess_video=False,
        )["frames"]
        if len(trajectory) != int(self.config["rollout"]["reverse_denoise_steps"]) + 1:
            raise RuntimeError("pipeline did not expose the complete reverse trajectory")
        if prepared.memory_latents is None:
            prepared.memory_latents = captured_condition["conditioning_latents"].clone()
            prepared.prompt_embeds = captured_condition["prompt_embeds"].clone()
            prepared.condition_template = GeSimConditionBatch(
                memory_latents=prepared.memory_latents,
                prompt_embeds=prepared.prompt_embeds,
                cond_to_concat=captured_condition["cond_to_concat"],
                condition_indicator=captured_condition["cond_indicator"],
                condition_mask=captured_condition["cond_mask"],
                padding_mask=captured_condition["padding_mask"],
                fps=16, n_view=views, n_previous=int(self.config["rollout"]["history_frames"]),
                num_future_latent_frames=trajectory[-1]["latents"].shape[2],
            )
        torch.save({"chunks": [trajectory], "num_chunks": 1}, seed_dir / "trajectory.pt")
        torch.save(trajectory[-1]["latents"], seed_dir / "final_future_latent.pt")
        # The condition is identical for every seed in a group and can be very
        # large, so store one shared copy instead of duplicating it per seed.
        condition_path = seed_dir.parent / "condition.pt"
        if not condition_path.exists():
            torch.save(prepared.condition_template, condition_path)
        videos = torch.cat((observation, result.detach().cpu()), dim=2).clamp(-1, 1)
        for view, camera in enumerate(self.args.data["train"]["valid_cam"]):
            save_video(videos[view], str(seed_dir / f"{camera}_color.mp4"), fps=16)
        rollout_metadata = {
            "condition_id": prepared.condition_id, "policy_version": self.policy_version,
            "seed": int(seed), "num_chunks": 1,
        }
        (seed_dir / "rollout.json").write_text(json.dumps(rollout_metadata, indent=2), encoding="utf-8")

    def set_policy_version(self, version: int) -> None:
        if version < self.policy_version:
            raise ValueError("policy version cannot move backwards")
        self.policy_version = int(version)
