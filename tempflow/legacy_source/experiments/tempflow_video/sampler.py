from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch

from experiments.awm_coca.gesim_runtime import PersistentGeSimRuntime, PreparedGeSimCondition
from experiments.tempflow_video.dynamics import noise_aware_weights
from experiments.tempflow_video.policy import VideoPolicyAdapter
from experiments.tempflow_video.schemas import (
    BranchGroupKey,
    BranchRollout,
    CollectedTransition,
    OrdinaryGroupKey,
    OrdinaryRollout,
)
from tempflow_video.runtime.rng_isolation import isolated_rng
from utils import save_video


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [torch.cuda.current_device() if device.index is None else int(device.index)]


class VideoTrajectorySampler:
    """Capture GE-Sim's deterministic base trajectory for a fixed initial seed."""

    def __init__(self, runtime: PersistentGeSimRuntime) -> None:
        self.runtime = runtime

    def sample_base(
        self,
        prepared: PreparedGeSimCondition,
        *,
        initial_seed: int,
        output_dir: str | Path,
        prompt: str,
    ):
        with isolated_rng(
            int(initial_seed), devices=_rng_devices(self.runtime.device)
        ):
            _, artifacts = self.runtime.rollout_group(
                prepared,
                seeds=[int(initial_seed)],
                output_dir=output_dir,
                prompt=prompt,
                expected_group_size=1,
                rollout_batch_size=1,
            )
        if len(artifacts) != 1:
            raise RuntimeError("deterministic base rollout must produce exactly one artifact")
        return artifacts[0]


class TempFlowBranchSampler:
    """ODE prefix -> one SDE transition -> ODE suffix in diffusion time."""

    def __init__(
        self,
        runtime: PersistentGeSimRuntime,
        *,
        eta: float,
        noise_aware_weighting: bool,
        noise_weight_normalization: str = "schedule_mean",
    ) -> None:
        self.runtime = runtime
        self.policy = VideoPolicyAdapter(runtime)
        self.trajectory_sampler = VideoTrajectorySampler(runtime)
        self.eta = float(eta)
        self.noise_aware_weighting = bool(noise_aware_weighting)
        self.noise_weight_normalization = str(noise_weight_normalization)

    def _flow_times(self) -> list[float]:
        sigmas = torch.as_tensor(self.runtime.scheduler.sigmas, dtype=torch.float64).flatten()
        return [float((sigma / (sigma + 1.0)).item()) for sigma in sigmas]

    def sample_base(
        self,
        prepared: PreparedGeSimCondition,
        *,
        initial_seed: int,
        output_dir: str | Path,
        prompt: str,
    ):
        """Collect one deterministic ODE path for all timestep branches."""

        return self.trajectory_sampler.sample_base(
            prepared,
            initial_seed=initial_seed,
            output_dir=Path(output_dir) / "base_trajectories",
            prompt=prompt,
        )

    def resolve_branch_timesteps(
        self,
        *,
        configured: Sequence[int] | None = None,
        timestep_fraction: float = 0.99,
    ) -> list[int]:
        """Resolve the official prefix-of-schedule timestep selection.

        TempFlow trains ``int(num_steps * timestep_fraction)`` leading steps
        and its per-step sampler omits the final scheduler transition.  GE-Sim
        additionally carries a duplicate terminal sigma; zero-length pairs are
        never legal SDE actions.
        """

        flow_times = self._flow_times()
        num_steps = len(flow_times) - 1
        if configured is not None:
            selected = [int(value) for value in configured]
        else:
            fraction = float(timestep_fraction)
            if not 0.0 < fraction <= 1.0:
                raise ValueError("timestep_fraction must lie in (0, 1]")
            selected = [
                index
                for index in range(int(num_steps * fraction))
                if flow_times[index + 1] < flow_times[index]
            ]
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("branch timesteps must be non-empty and unique")
        for index in selected:
            if not 0 <= index < num_steps:
                raise ValueError(f"branch timestep {index} lies outside [0, {num_steps - 1}]")
            if not flow_times[index + 1] < flow_times[index]:
                raise ValueError(
                    f"branch timestep {index} is a zero/non-reverse scheduler transition"
                )
        return selected

    def resolve_branch_timestep_fractions(self, fractions: Sequence[float]) -> list[int]:
        """Pick legal reverse transitions nearest requested schedule positions."""
        flow_times = self._flow_times()
        legal = self.resolve_branch_timesteps(configured=None, timestep_fraction=1.0)
        start, end = float(flow_times[0]), float(flow_times[-1])
        span = start - end
        if span <= 0.0:
            raise ValueError("scheduler flow times do not define a reverse interval")
        selected = []
        for fraction in fractions:
            fraction = float(fraction)
            if not 0.0 < fraction < 1.0:
                raise ValueError("branch timestep fractions must lie in (0, 1)")
            selected.append(min(
                legal,
                key=lambda index: abs(((start - float(flow_times[index])) / span) - fraction),
            ))
        if len(selected) != len(set(selected)):
            raise ValueError("requested timestep fractions resolved to duplicate transitions")
        return selected

    @torch.inference_mode()
    def sample_group(
        self,
        prepared: PreparedGeSimCondition,
        *,
        initial_seed: int,
        branch_timestep: int,
        branch_noise_seeds: Sequence[int],
        branch_ids: Sequence[int] | None = None,
        output_dir: str | Path,
        prompt: str,
        prompt_id: str,
        reward_config_sha256: str,
        video_length: int = 29,
        base_artifact=None,
    ) -> tuple[Path, list[BranchRollout]]:
        if not branch_noise_seeds or len(set(map(int, branch_noise_seeds))) != len(branch_noise_seeds):
            raise ValueError("local branch shard requires unique branch noise seeds")
        if branch_ids is None:
            branch_ids = list(range(len(branch_noise_seeds)))
        if len(branch_ids) != len(branch_noise_seeds) or len(set(map(int, branch_ids))) != len(branch_ids):
            raise ValueError("branch_ids must be unique and match branch_noise_seeds")
        base = base_artifact
        if base is None:
            base = self.sample_base(
                prepared,
                initial_seed=initial_seed,
                output_dir=output_dir,
                prompt=prompt,
            )
        flow_times = self._flow_times()
        num_steps = len(base.trajectory) - 1
        if len(flow_times) != num_steps + 1:
            raise RuntimeError(
                f"scheduler/trajectory mismatch: {len(flow_times) - 1} transitions != {num_steps} steps"
            )
        if not 0 <= int(branch_timestep) < num_steps:
            raise ValueError(f"branch_timestep must lie in [0, {num_steps - 1}]")
        branch_timestep = int(branch_timestep)
        t = flow_times[branch_timestep]
        next_t = flow_times[branch_timestep + 1]
        if not next_t < t:
            raise ValueError(
                f"branch timestep {branch_timestep} has zero/non-reverse step: t={t}, next_t={next_t}"
            )
        weights = noise_aware_weights(
            flow_times,
            eta=self.eta,
            enabled=self.noise_aware_weighting,
            normalization=self.noise_weight_normalization,
        )
        group_key = BranchGroupKey(
            condition_id=prepared.condition_id,
            prompt_id=prompt_id,
            initial_seed=int(initial_seed),
            branch_timestep=branch_timestep,
            video_length=int(video_length),
            reward_config_sha256=reward_config_sha256,
        )
        group_dir = Path(output_dir) / "branch_rollouts" / group_key.group_id(self.runtime.policy_version)
        group_dir.mkdir(parents=True, exist_ok=False)
        current = base.trajectory[branch_timestep]["latents"].to(self.runtime.device)
        condition = base.condition_template
        prefix_digest = _tensor_sha256(current)
        prefix_trajectory_digests = [
            _tensor_sha256(row["latents"])
            for row in base.trajectory[: branch_timestep + 1]
        ]
        model_was_training = self.runtime.transformer.training
        self.runtime.transformer.eval()
        rollouts: list[BranchRollout] = []
        try:
            for branch_id, noise_seed in zip(branch_ids, branch_noise_seeds):
                branch_id = int(branch_id)
                with isolated_rng(
                    int(noise_seed), devices=_rng_devices(self.runtime.device)
                ):
                    generator = torch.Generator(device=self.runtime.device).manual_seed(int(noise_seed))
                    transition = self.policy.sample_one_step(
                        current,
                        condition,
                        flow_time=t,
                        next_flow_time=next_t,
                        stochastic=True,
                        eta=self.eta,
                        generator=generator,
                    )
                    latent = transition.next_sample
                    for step in range(branch_timestep + 1, num_steps):
                        step_t = flow_times[step]
                        step_next_t = flow_times[step + 1]
                        if step_next_t == step_t:
                            continue
                        latent = self.policy.sample_one_step(
                            latent,
                            condition,
                            flow_time=step_t,
                            next_flow_time=step_next_t,
                            stochastic=False,
                            eta=self.eta,
                        )
                    future = self.policy.decode_video(latent).detach().cpu()
                full_video = torch.cat((prepared.observation, future), dim=2).clamp(-1.0, 1.0)
                seed_dir = group_dir / f"branch_{branch_id:03d}"
                seed_dir.mkdir(parents=True, exist_ok=False)
                for view, camera in enumerate(self.runtime.args.data["train"]["valid_cam"]):
                    save_video(
                        full_video[view],
                        str(seed_dir / f"{camera}_color.mp4"),
                        fps=int(self.runtime.config["reward"].get("generated_fps", 30)),
                    )
                sample_id = (
                    f"{prepared.condition_id}:p{self.runtime.policy_version}:s{int(initial_seed)}:"
                    f"t{branch_timestep}:b{branch_id}"
                )
                rollout = BranchRollout(
                    sample_id=sample_id,
                    group_key=group_key,
                    policy_version=self.runtime.policy_version,
                    branch_id=branch_id,
                    branch_noise_seed=int(noise_seed),
                    seed_dir=seed_dir,
                    current_latent=current.detach().cpu(),
                    next_latent=transition.next_sample.detach().cpu(),
                    condition=condition,
                    flow_time=t,
                    next_flow_time=next_t,
                    eta=self.eta,
                    old_log_prob=float(transition.log_prob.mean().item()),
                    old_token_log_prob=transition.token_log_prob.detach().cpu(),
                    rf_noise_std=float(transition.rf_noise_std.item()),
                    noise_weight=float(weights[branch_timestep].item()),
                    prefix_latent_sha256=prefix_digest,
                    branch_noise_sha256=_tensor_sha256(transition.exploration_noise),
                )
                (seed_dir / "rollout.json").write_text(
                    json.dumps(rollout.metadata(), ensure_ascii=False, indent=2), encoding="utf-8"
                )
                rollouts.append(rollout)
        finally:
            self.runtime.transformer.train(model_was_training)
        group_payload = {
            "group_id": group_key.group_id(self.runtime.policy_version),
            "group_key": group_key.__dict__,
            "policy_version": self.runtime.policy_version,
            "branch_factor": len(rollouts),
            "prefix_latent_sha256": prefix_digest,
            "prefix_trajectory_sha256": prefix_trajectory_digests,
            "branch_noise_sha256": [rollout.branch_noise_sha256 for rollout in rollouts],
            "video_sha256": {
                camera: [
                    _file_sha256(rollout.seed_dir / f"{camera}_color.mp4")
                    for rollout in rollouts
                ]
                for camera in self.runtime.args.data["train"]["valid_cam"]
            },
        }
        if len(rollouts) > 1 and all(
            len(set(values)) == 1 for values in group_payload["video_sha256"].values()
        ):
            raise RuntimeError("all branch videos are byte-identical despite distinct exploration noise")
        (group_dir / "group.json").write_text(
            json.dumps(group_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return group_dir, rollouts


class FlowGRPOVideoSampler:
    """Ordinary video GRPO collector with an SDE action at every legal step.

    Group members use independent initial noise seeds; unlike TempFlow branching
    they do not share a prefix. Terminal video reward is assigned to every
    transition on that member's stochastic trajectory.
    """

    def __init__(
        self,
        runtime: PersistentGeSimRuntime,
        *,
        eta: float,
        noise_aware_weighting: bool,
        noise_weight_normalization: str = "schedule_mean",
    ) -> None:
        self.runtime = runtime
        self.policy = VideoPolicyAdapter(runtime)
        self.eta = float(eta)
        self.noise_aware_weighting = bool(noise_aware_weighting)
        self.noise_weight_normalization = str(noise_weight_normalization)

    def _flow_times(self) -> list[float]:
        sigmas = torch.as_tensor(self.runtime.scheduler.sigmas, dtype=torch.float64).flatten()
        return [float((sigma / (sigma + 1.0)).item()) for sigma in sigmas]

    @torch.inference_mode()
    def sample_group(
        self,
        prepared: PreparedGeSimCondition,
        *,
        initial_seeds: Sequence[int],
        transition_noise_seed_base: int,
        group_sequence: int,
        output_dir: str | Path,
        prompt: str,
        prompt_id: str,
        reward_config_sha256: str,
        video_length: int = 29,
    ) -> tuple[Path, list[OrdinaryRollout]]:
        if len(initial_seeds) < 2 or len(set(map(int, initial_seeds))) != len(initial_seeds):
            raise ValueError("ordinary GRPO requires at least two unique initial seeds")
        _, base_artifacts = self.runtime.rollout_group(
            prepared,
            seeds=[int(seed) for seed in initial_seeds],
            output_dir=Path(output_dir) / "ordinary_initial_trajectories",
            prompt=prompt,
            expected_group_size=len(initial_seeds),
            # One batch gives every group member the same conditioning-cache
            # state; chunking the first member uncached and later members cached
            # would make seed semantics depend on rollout position.
            rollout_batch_size=len(initial_seeds),
        )
        flow_times = self._flow_times()
        if any(len(artifact.trajectory) != len(flow_times) for artifact in base_artifacts):
            raise RuntimeError("scheduler/ordinary trajectory length mismatch")
        weights = noise_aware_weights(
            flow_times,
            eta=self.eta,
            enabled=self.noise_aware_weighting,
            normalization=self.noise_weight_normalization,
        )
        key = OrdinaryGroupKey(
            condition_id=prepared.condition_id,
            prompt_id=prompt_id,
            video_length=int(video_length),
            reward_config_sha256=reward_config_sha256,
            group_sequence=int(group_sequence),
        )
        group_dir = Path(output_dir) / "ordinary_rollouts" / key.group_id(self.runtime.policy_version)
        group_dir.mkdir(parents=True, exist_ok=False)
        model_was_training = self.runtime.transformer.training
        self.runtime.transformer.eval()
        output: list[OrdinaryRollout] = []
        try:
            for rollout_id, (initial_seed, artifact) in enumerate(zip(initial_seeds, base_artifacts)):
                latent = artifact.trajectory[0]["latents"].to(self.runtime.device)
                transitions = []
                for timestep, (flow_time, next_flow_time) in enumerate(
                    zip(flow_times[:-1], flow_times[1:])
                ):
                    if next_flow_time == flow_time:
                        continue
                    noise_seed = int(transition_noise_seed_base) + rollout_id * 10_000 + timestep
                    generator = torch.Generator(device=self.runtime.device).manual_seed(noise_seed)
                    transition = self.policy.sample_one_step(
                        latent,
                        artifact.condition_template,
                        flow_time=flow_time,
                        next_flow_time=next_flow_time,
                        stochastic=True,
                        eta=self.eta,
                        generator=generator,
                    )
                    transitions.append(
                        CollectedTransition(
                            timestep=timestep,
                            current_latent=latent.detach().cpu(),
                            next_latent=transition.next_sample.detach().cpu(),
                            flow_time=flow_time,
                            next_flow_time=next_flow_time,
                            eta=self.eta,
                            old_log_prob=float(transition.log_prob.mean().item()),
                            old_token_log_prob=transition.token_log_prob.detach().cpu(),
                            rf_noise_std=float(transition.rf_noise_std.item()),
                            noise_weight=float(weights[timestep].item()),
                            noise_seed=noise_seed,
                        )
                    )
                    latent = transition.next_sample
                if not transitions:
                    raise RuntimeError("ordinary SDE trajectory did not collect any transition")
                future = self.policy.decode_video(latent).detach().cpu()
                full_video = torch.cat((prepared.observation, future), dim=2).clamp(-1.0, 1.0)
                seed_dir = group_dir / f"rollout_{rollout_id:03d}"
                seed_dir.mkdir(parents=True, exist_ok=False)
                for view, camera in enumerate(self.runtime.args.data["train"]["valid_cam"]):
                    save_video(
                        full_video[view],
                        str(seed_dir / f"{camera}_color.mp4"),
                        fps=int(self.runtime.config["reward"].get("generated_fps", 30)),
                    )
                rollout = OrdinaryRollout(
                    sample_id=(
                        f"{prepared.condition_id}:p{self.runtime.policy_version}:"
                        f"ordinary{int(group_sequence)}:r{rollout_id}"
                    ),
                    group_key=key,
                    policy_version=self.runtime.policy_version,
                    rollout_id=rollout_id,
                    initial_seed=int(initial_seed),
                    seed_dir=seed_dir,
                    condition=artifact.condition_template,
                    transitions=transitions,
                )
                (seed_dir / "rollout.json").write_text(
                    json.dumps(rollout.metadata(), ensure_ascii=False, indent=2), encoding="utf-8"
                )
                output.append(rollout)
        finally:
            self.runtime.transformer.train(model_was_training)
        (group_dir / "group.json").write_text(
            json.dumps(
                {
                    "group_id": key.group_id(self.runtime.policy_version),
                    "group_key": key.__dict__,
                    "policy_version": self.runtime.policy_version,
                    "group_size": len(output),
                    "initial_seeds": [int(seed) for seed in initial_seeds],
                    "transitions_per_rollout": [len(item.transitions) for item in output],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return group_dir, output
