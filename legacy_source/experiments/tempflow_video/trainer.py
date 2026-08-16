from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch

from experiments.tempflow_video.dynamics import edm_sde_transition_with_logprob, edm_transition_mean
from experiments.tempflow_video.loss import tempflow_grpo_loss
from experiments.tempflow_video.policy import ReferencePolicyAdapter, VideoPolicyAdapter
from experiments.tempflow_video.schemas import BranchRollout, OrdinaryRollout, TrainerStepRecord


def _gradient_norm(values: Iterable[torch.Tensor | None]) -> float:
    squared = 0.0
    for value in values:
        if value is not None:
            squared += float(value.detach().float().square().sum().item())
    return math.sqrt(squared)


def _add_gradients(
    buffers: Sequence[torch.Tensor], gradients: Sequence[torch.Tensor | None], *, scale: float
) -> None:
    for buffer, gradient in zip(buffers, gradients):
        if gradient is not None:
            buffer.add_(gradient.detach(), alpha=scale)


@dataclass(frozen=True)
class TempFlowOptimizerConfig:
    learning_rate: float = 1.0e-6
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1.0e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    clip_range: float = 1.0e-4
    kl_beta: float = 0.01
    warmup_steps: int = 0
    log_term_grad_norm: bool = True
    log_gradient_cosine: bool = False


class TempFlowVideoTrainer:
    """TempFlow-GRPO optimizer with explicit old-policy rollout buffers."""

    def __init__(
        self,
        policy: VideoPolicyAdapter,
        reference: ReferencePolicyAdapter,
        config: TempFlowOptimizerConfig,
        gradient_reducer: Callable[[Sequence[torch.Tensor]], None] | None = None,
    ) -> None:
        self.policy = policy
        self.reference = reference
        self.config = config
        self.gradient_reducer = gradient_reducer
        self.parameters = policy.get_trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=float(config.learning_rate),
            betas=tuple(config.betas),
            eps=float(config.epsilon),
            weight_decay=float(config.weight_decay),
        )
        warmup = max(int(config.warmup_steps), 0)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: min(1.0, float(step + 1) / max(warmup, 1)) if warmup else 1.0,
        )
        self.optimizer_step = 0
        self.policy_version = 0
        self.group_attempts = 0
        self.rollout_epoch = 0
        self._consumed_groups: set[str] = set()
        self._previous_gradients: list[torch.Tensor] | None = None

    def _validate_group(
        self,
        rollouts: Sequence[BranchRollout | OrdinaryRollout],
        *,
        global_group_size: int,
    ) -> str:
        if not rollouts:
            raise ValueError("every distributed rank needs at least one local branch rollout")
        if global_group_size < 2 or len(rollouts) > global_group_size:
            raise ValueError("TempFlow update requires at least two global branch rollouts")
        group_ids = {item.group_key.group_id(item.policy_version) for item in rollouts}
        if len(group_ids) != 1:
            raise ValueError("advantage group mixes incompatible rollout metadata")
        group_id = next(iter(group_ids))
        if group_id in self._consumed_groups:
            raise ValueError(f"rollout group already consumed: {group_id}")
        if {item.policy_version for item in rollouts} != {self.policy_version}:
            raise ValueError(
                f"stale rollout group: expected policy_version={self.policy_version}"
            )
        if any(item.advantage is None or item.reward is None for item in rollouts):
            raise ValueError("every rollout needs terminal reward and group advantage")
        if not all(math.isfinite(float(item.advantage)) for item in rollouts):
            raise ValueError("advantages must be finite")
        return group_id

    def update_group(
        self,
        rollouts: Sequence[BranchRollout | OrdinaryRollout],
        *,
        global_group_size: int | None = None,
    ) -> TrainerStepRecord:
        global_count = int(global_group_size or len(rollouts))
        group_id = self._validate_group(rollouts, global_group_size=global_count)
        record = self._update_rollouts(rollouts, global_rollout_count=global_count)
        self._consumed_groups.add(group_id)
        return record

    def _update_rollouts(
        self,
        rollouts: Sequence[BranchRollout | OrdinaryRollout],
        *,
        global_rollout_count: int | None = None,
    ) -> TrainerStepRecord:
        """Apply one optimizer step to a minibatch already validated as on-policy.

        A minibatch may contain several independently standardized timestep
        groups collected by the same frozen old policy.  This mirrors the
        official TempFlow sampling-then-training data flow: later minibatches
        retain their collection-time ``old_log_prob`` after earlier minibatches
        have changed the current policy.
        """

        if not rollouts:
            raise ValueError("TempFlow minibatch must not be empty")
        global_count = int(global_rollout_count or len(rollouts))
        self.reference.assert_unchanged()
        self.policy.policy_model.train()
        trainable_before = [parameter.detach().clone() for parameter in self.parameters]
        self.optimizer.zero_grad(set_to_none=True)
        term_buffers = None
        if self.config.log_term_grad_norm:
            term_buffers = {
                name: [torch.zeros_like(parameter) for parameter in self.parameters]
                for name in ("policy", "raw_kl", "weighted_kl")
            }
        metric_rows: list[dict[str, float]] = []
        for rollout in rollouts:
            actions = rollout.transitions if isinstance(rollout, OrdinaryRollout) else [rollout]
            # Every rank contributes its local shard to one global group mean.
            # The reducer performs SUM, so no extra world-size division belongs here.
            action_scale = 1.0 / (global_count * len(actions))
            for action in actions:
                current = action.current_latent.to(self.policy.runtime.device)
                collected_next = action.next_latent.to(self.policy.runtime.device)
                velocity = self.policy.predict_velocity_or_noise(
                    current, action.flow_time, rollout.condition
                )
                transition = edm_sde_transition_with_logprob(
                    current,
                    velocity,
                    flow_time=action.flow_time,
                    next_flow_time=action.next_flow_time,
                    eta=action.eta,
                    next_sample=collected_next,
                )
                with torch.no_grad():
                    reference_velocity = self.reference.predict_velocity_or_noise(
                        current, action.flow_time, rollout.condition
                    )
                    reference_mean, _, _ = edm_transition_mean(
                        current,
                        reference_velocity,
                        flow_time=action.flow_time,
                        next_flow_time=action.next_flow_time,
                        eta=action.eta,
                    )
                output = tempflow_grpo_loss(
                    log_probs=transition.log_prob.mean().reshape(1),
                    old_log_probs=torch.tensor(
                        [action.old_log_prob], device=current.device, dtype=torch.float32
                    ),
                    advantages=torch.tensor(
                        [float(rollout.advantage)], device=current.device, dtype=torch.float32
                    ),
                    noise_weights=torch.tensor(
                        [action.noise_weight], device=current.device, dtype=torch.float32
                    ),
                    policy_means=transition.mean.unsqueeze(0),
                    reference_means=reference_mean.unsqueeze(0),
                    transition_stds=transition.std.reshape(1),
                    clip_range=self.config.clip_range,
                    kl_beta=self.config.kl_beta,
                )
                if term_buffers is not None:
                    terms = {
                        "policy": output.policy_loss,
                        "raw_kl": output.raw_kl_loss,
                        "weighted_kl": output.weighted_kl_loss,
                    }
                    for name, term in terms.items():
                        gradients = torch.autograd.grad(
                            term,
                            self.parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        _add_gradients(term_buffers[name], gradients, scale=action_scale)
                (output.total_loss * action_scale).backward()
                row = output.detached_metrics()
                row.update(
                    {
                        "reward": float(
                            rollout.reward.get("training_reward", rollout.reward["total_reward"])
                        ),
                        "advantage": float(rollout.advantage),
                        "noise_weight": float(action.noise_weight),
                    }
                )
                metric_rows.append(row)

        if self.gradient_reducer is not None:
            gradients = []
            for parameter in self.parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                gradients.append(parameter.grad)
            self.gradient_reducer(gradients)
            if term_buffers is not None:
                for buffers in term_buffers.values():
                    self.gradient_reducer(buffers)

        total_grad_norm_before = _gradient_norm(parameter.grad for parameter in self.parameters)
        if not math.isfinite(total_grad_norm_before):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite TempFlow gradient")
        gradient_cosine = 0.0
        has_previous_gradient = 0.0
        if self.config.log_gradient_cosine:
            current_gradients = [
                (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad)
                .detach()
                .float()
                .cpu()
                for parameter in self.parameters
            ]
            if self._previous_gradients is not None:
                dot = sum(
                    float((current * previous).sum().item())
                    for current, previous in zip(current_gradients, self._previous_gradients)
                )
                current_norm = _gradient_norm(current_gradients)
                previous_norm = _gradient_norm(self._previous_gradients)
                if current_norm > 0.0 and previous_norm > 0.0:
                    gradient_cosine = dot / (current_norm * previous_norm)
                    has_previous_gradient = 1.0
            self._previous_gradients = current_gradients
        clip_result = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm)
        if not torch.isfinite(clip_result):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite TempFlow clipped gradient")
        total_grad_norm_after = _gradient_norm(parameter.grad for parameter in self.parameters)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_step += 1
        self.policy_version += 1
        self.reference.assert_unchanged()
        parameter_delta_squared = sum(
            float((parameter.detach() - before).float().square().sum().item())
            for parameter, before in zip(self.parameters, trainable_before)
        )
        changed_trainable_count = sum(
            not torch.equal(parameter.detach(), before)
            for parameter, before in zip(self.parameters, trainable_before)
        )
        if changed_trainable_count == 0:
            raise RuntimeError("optimizer step did not mutate any trainable policy parameter")

        keys = metric_rows[0].keys()
        metric_count = len(metric_rows)
        metrics = {key: float(sum(row[key] for row in metric_rows) / metric_count) for key in keys}
        metrics.update(
            {
                "policy_grad_norm": 0.0
                if term_buffers is None
                else _gradient_norm(term_buffers["policy"]),
                "raw_kl_grad_norm": 0.0
                if term_buffers is None
                else _gradient_norm(term_buffers["raw_kl"]),
                "weighted_kl_grad_norm": 0.0
                if term_buffers is None
                else _gradient_norm(term_buffers["weighted_kl"]),
                "total_grad_norm_before_clip": total_grad_norm_before,
                "total_grad_norm_after_clip": total_grad_norm_after,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "changed_trainable_parameter_tensors": float(changed_trainable_count),
                "parameter_delta_norm": math.sqrt(parameter_delta_squared),
                "gradient_cosine_with_previous_step": float(gradient_cosine),
                "gradient_cosine_has_previous_step": has_previous_gradient,
            }
        )
        return TrainerStepRecord(
            optimizer_step=self.optimizer_step,
            policy_version=self.policy_version,
            metrics=metrics,
        )

    def update_rollout_buffer(
        self,
        groups: Sequence[Sequence[BranchRollout | OrdinaryRollout]],
        *,
        minibatches_per_epoch: int = 2,
        num_inner_epochs: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        max_updates: int | None = None,
        global_group_sizes: Sequence[int] | None = None,
    ) -> list[TrainerStepRecord]:
        """Train on groups collected under one frozen policy snapshot.

        Group-relative advantages remain local to each branch timestep, while
        optimizer minibatches may combine several groups.  All groups are
        validated before the first parameter mutation.  The whole buffer is
        discarded afterwards, including when ``max_updates`` truncates the
        final epoch, because it is stale once any update has occurred.
        """

        if not groups:
            raise ValueError("rollout buffer must contain at least one group")
        if minibatches_per_epoch <= 0 or num_inner_epochs <= 0:
            raise ValueError("minibatches_per_epoch and num_inner_epochs must be positive")
        if max_updates is not None and max_updates <= 0:
            raise ValueError("max_updates must be positive when provided")

        collection_policy_version = self.policy_version
        if global_group_sizes is None:
            global_group_sizes = [len(group) for group in groups]
        if len(global_group_sizes) != len(groups):
            raise ValueError("global_group_sizes must match rollout buffer groups")
        group_ids: list[str] = []
        normalized_groups: list[tuple[list[BranchRollout | OrdinaryRollout], int]] = []
        for group, global_size in zip(groups, global_group_sizes):
            group_id = self._validate_group(group, global_group_size=int(global_size))
            if group_id in group_ids:
                raise ValueError(f"rollout buffer repeats group: {group_id}")
            if {item.policy_version for item in group} != {collection_policy_version}:
                raise ValueError("rollout buffer mixes old-policy snapshots")
            group_ids.append(group_id)
            normalized_groups.append((list(group), int(global_size)))

        records: list[TrainerStepRecord] = []
        rng = random.Random(int(seed))
        try:
            for _ in range(int(num_inner_epochs)):
                ordered = list(normalized_groups)
                if shuffle:
                    rng.shuffle(ordered)
                batch_count = min(int(minibatches_per_epoch), len(ordered))
                # Contiguous balanced chunks preserve every group exactly once
                # per inner epoch without splitting its six sibling branches.
                for batch_index in range(batch_count):
                    start = batch_index * len(ordered) // batch_count
                    stop = (batch_index + 1) * len(ordered) // batch_count
                    selected = ordered[start:stop]
                    minibatch = [rollout for group, _ in selected for rollout in group]
                    global_count = sum(global_size for _, global_size in selected)
                    records.append(
                        self._update_rollouts(
                            minibatch, global_rollout_count=global_count
                        )
                    )
                    if max_updates is not None and len(records) >= int(max_updates):
                        return records
        finally:
            # Once the current policy changes, no item from this old-policy
            # buffer may be silently reused in a later collection epoch.
            self._consumed_groups.update(group_ids)
        return records

    def state_dict(self) -> dict:
        return {
            "optimizer_step": self.optimizer_step,
            "policy_version": self.policy_version,
            "group_attempts": self.group_attempts,
            "rollout_epoch": self.rollout_epoch,
            "consumed_groups": sorted(self._consumed_groups),
        }

    def load_state_dict(self, state: dict) -> None:
        self.optimizer_step = int(state.get("optimizer_step", 0))
        self.policy_version = int(state.get("policy_version", self.optimizer_step))
        self.group_attempts = int(state.get("group_attempts", self.policy_version))
        self.rollout_epoch = int(state.get("rollout_epoch", 0))
        self._consumed_groups = set(state.get("consumed_groups", ()))
