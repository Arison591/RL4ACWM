from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist

from experiments.awm_coca.ema import ParameterEMA
from experiments.awm_coca.training_core import ProposalConfig, RolloutTrainingSample, awm_coca_sample_loss


def _synchronize_gradients(parameters: Sequence[torch.nn.Parameter]) -> None:
    """Average every trainable gradient in an identical collective order.

    A conditional path can leave a parameter unused on one rank.  Skipping its
    collective only on that rank makes the remaining NCCL calls mismatch and
    eventually time out.  A missing local gradient is a zero contribution, so
    materialize it before the all-reduce.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def _synchronize_tensors(tensors: Sequence[torch.Tensor]) -> None:
    """Average every tensor in an identical collective order (buffer variant).

    Unlike _synchronize_gradients, the buffers are never None (zeros for unused
    parameters), so no materialization is needed.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    for tensor in tensors:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(world_size)


def _grad_norm(tensors: Iterable[torch.Tensor | None]) -> float:
    """L2 norm of the concatenated gradients, matching clip_grad_norm_ (norm_type=2)."""
    squared = 0.0
    for tensor in tensors:
        if tensor is None:
            continue
        squared += float(tensor.square().sum().item())
    return math.sqrt(squared)


def _term_grad_norms(
    parameters: Sequence[torch.nn.Parameter],
    kl_buffers: Sequence[torch.Tensor],
) -> tuple[float, float]:
    """Split the accumulated total gradient into fm / reference-KL term norms.

    total = fm + kl, so fm_grad = param.grad - kl_buffer.  Each parameter's
    gradient is already all-reduced and averaged by the caller.
    """
    fm_squared = 0.0
    for parameter, kl_buffer in zip(parameters, kl_buffers):
        total_grad = parameter.grad
        if total_grad is None:
            total_grad = torch.zeros_like(parameter)
        fm_squared += float((total_grad - kl_buffer).square().sum().item())
    return math.sqrt(fm_squared), _grad_norm(kl_buffers)


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
    log_term_grad_norm: bool = True


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
        self._log_term_grad_norm = bool(optimizer_config.log_term_grad_norm)
        # reference-KL 项梯度的累加缓冲，与主梯度同尺度（scale / accum），用于逐项范数统计。
        self._kl_term_grad_accum: list[torch.Tensor] | None = (
            [torch.zeros_like(parameter) for parameter in self.parameters]
            if self._log_term_grad_norm
            else None
        )
        self.optimizer.zero_grad(set_to_none=True)

    def update_group(
        self,
        fresh_rollouts: Sequence[RolloutTrainingSample],
        *,
        group_id: str,
        generator: torch.Generator | None = None,
        normalization_count: int | None = None,
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
        local_count = len(fresh_rollouts)
        if normalization_count is not None:
            # 有效样本数是跨卡全局统计（leave-one-out advantage 也基于全局有效集）。
            # 为保证各卡样本最终权重一致（1/global_valid），本地缩放需乘 world_size，
            # 与下方 all_reduce SUM 后 /world_size 抵消。
            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            scale = float(world_size) / float(normalization_count)
        else:
            scale = 1.0 / local_count
        for sample in fresh_rollouts:
            sample_loss, record, fm_term, kl_term = awm_coca_sample_loss(
                self.adapter, sample, self.proposal_config, beta=self.config.beta, generator=generator
            )
            if not torch.isfinite(sample_loss):
                self.optimizer.zero_grad(set_to_none=True)
                self._reset_term_grad_accum()
                raise FloatingPointError(f"non-finite AWM-CoCA loss: {sample_loss.detach().item()}")
            scaled_loss = sample_loss * scale / self.config.gradient_accumulation_steps
            if self._kl_term_grad_accum is not None:
                self._accumulate_kl_gradient(kl_term, scale)
            scaled_loss.backward()
            detached_loss += float(sample_loss.detach().item()) / local_count
            records.append(record)
        self.accumulation_step += 1
        self._consumed_groups.add(group_id)
        stepped = self.accumulation_step == self.config.gradient_accumulation_steps
        grad_norm = None
        fm_grad_norm = None
        kl_grad_norm = None
        if stepped:
            _synchronize_gradients(self.parameters)
            if self._kl_term_grad_accum is not None:
                _synchronize_tensors(self._kl_term_grad_accum)
                fm_grad_norm, kl_grad_norm = _term_grad_norms(
                    self.parameters, self._kl_term_grad_accum
                )
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm)
            if not torch.isfinite(grad_norm_tensor):
                self.optimizer.zero_grad(set_to_none=True)
                self._reset_term_grad_accum()
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
            self._reset_term_grad_accum()
        return {
            "optimizer_step": self.optimizer_step,
            "policy_version": self.policy_version,
            "optimizer_stepped": stepped,
            "accumulation_step": self.accumulation_step,
            "loss": detached_loss,
            "grad_norm": grad_norm,
            "fm_grad_norm": fm_grad_norm,
            "kl_grad_norm": kl_grad_norm,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "samples": [record.to_dict() for record in records],
        }

    def _accumulate_kl_gradient(self, kl_term: torch.Tensor, scale: float) -> None:
        """Accumulate the reference-KL term's gradient for per-term norm logging.

        Uses autograd.grad with retain_graph=True so the main scaled_loss.backward()
        in update_group still works on the shared graph.  The buffer is scaled by
        scale / accumulation_steps, identical to the main gradient accumulation, so
        the reported per-term norms are directly comparable to trainer/grad_norm.
        """
        factor = scale / self.config.gradient_accumulation_steps
        grads = torch.autograd.grad(
            kl_term,
            self.parameters,
            retain_graph=True,
            allow_unused=True,
        )
        for buffer, grad in zip(self._kl_term_grad_accum, grads):
            if grad is not None:
                buffer.add_(grad, alpha=factor)

    def _reset_term_grad_accum(self) -> None:
        """Zero the per-term accumulation buffer (at step and on abort paths)."""
        if self._kl_term_grad_accum is not None:
            for buffer in self._kl_term_grad_accum:
                buffer.zero_()

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
