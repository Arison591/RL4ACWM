from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist

from experiments.awm_coca.ema import ParameterEMA
from experiments.awm_coca.training_core import ProposalConfig, RolloutTrainingSample, awm_coca_sample_loss


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    beta: float = 0.01
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100


class AWMCoCATrainer:
    """One-use, fresh-policy AWM-CoCA optimizer.

    Accumulation may span conditions, but every accumulated group must have the
    same policy version. The policy version increments only after optimizer.step.
    """

    def __init__(
        self,
        adapter: Any,
        proposal_config: ProposalConfig,
        optimizer_config: OptimizerConfig,
        *,
        parameters: Iterable[torch.nn.Parameter] | None = None,
        ema: ParameterEMA | None = None,
    ) -> None:
        self.adapter = adapter
        self.proposal_config = proposal_config
        self.config = optimizer_config
        self.parameters = list(parameters or (p for p in adapter.policy_model.parameters() if p.requires_grad))
        if not self.parameters:
            raise ValueError("policy model has no trainable parameters")
        if optimizer_config.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.epsilon,
            weight_decay=optimizer_config.weight_decay,
        )
        warmup = max(optimizer_config.warmup_steps, 0)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(1.0, float(step + 1) / max(warmup, 1)) if warmup else 1.0
        )
        self.ema = ema
        self.optimizer_step = 0
        self.policy_version = 0
        self.accumulation_step = 0
        self._consumed_groups: set[str] = set()
        self.optimizer.zero_grad(set_to_none=True)

    def update_group(
        self,
        fresh_rollouts: Sequence[RolloutTrainingSample],
        *,
        group_id: str,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        if not group_id or group_id in self._consumed_groups:
            raise ValueError(f"rollout group is missing or already consumed: {group_id!r}")
        versions = {sample.policy_version for sample in fresh_rollouts}
        if versions != {self.policy_version}:
            raise ValueError(f"expected policy_version={self.policy_version}, got {sorted(versions)}")
        self.adapter.policy_model.train()
        if not fresh_rollouts:
            raise ValueError("fresh rollout group must not be empty")
        records = []
        detached_loss = 0.0
        group_size = len(fresh_rollouts)
        for sample in fresh_rollouts:
            sample_loss, record = awm_coca_sample_loss(
                self.adapter, sample, self.proposal_config, beta=self.config.beta, generator=generator
            )
            if not torch.isfinite(sample_loss):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"non-finite AWM-CoCA loss: {sample_loss.detach().item()}")
            scaled_loss = sample_loss / (group_size * self.config.gradient_accumulation_steps)
            scaled_loss.backward()
            detached_loss += float(sample_loss.detach().item()) / group_size
            records.append(record)
        self.accumulation_step += 1
        self._consumed_groups.add(group_id)
        stepped = self.accumulation_step == self.config.gradient_accumulation_steps
        grad_norm = None
        if stepped:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                for parameter in self.parameters:
                    if parameter.grad is None:
                        continue
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad.div_(world_size)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm)
            if not torch.isfinite(grad_norm_tensor):
                self.optimizer.zero_grad(set_to_none=True)
                self.accumulation_step = 0
                raise FloatingPointError(f"non-finite gradient norm: {grad_norm_tensor.detach().item()}")
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.adapter.policy_model.named_parameters())
            self.optimizer_step += 1
            self.policy_version += 1
            self.accumulation_step = 0
            grad_norm = float(grad_norm_tensor.detach().item())
        return {
            "optimizer_step": self.optimizer_step,
            "policy_version": self.policy_version,
            "optimizer_stepped": stepped,
            "accumulation_step": self.accumulation_step,
            "loss": detached_loss,
            "grad_norm": grad_norm,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "samples": [record.to_dict() for record in records],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer_step": self.optimizer_step,
            "policy_version": self.policy_version,
            "accumulation_step": self.accumulation_step,
            "consumed_groups": sorted(self._consumed_groups),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("accumulation_step", 0)) != 0:
            raise ValueError("checkpoints taken during gradient accumulation are not resumable")
        self.optimizer_step = int(state.get("optimizer_step", 0))
        self.policy_version = int(state.get("policy_version", self.optimizer_step))
        self.accumulation_step = 0
        self._consumed_groups = set(state.get("consumed_groups", ()))
