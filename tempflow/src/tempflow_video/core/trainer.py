from __future__ import annotations

import math

import torch

from .policy_objective import ComponentPolicyLoss, gradient_diagnostics


def grad_norm(parameters) -> float:
    return math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in parameters if p.grad is not None))


def optimizer_step(loss: ComponentPolicyLoss, *, parameters: list[torch.nn.Parameter],
                   optimizer: torch.optim.Optimizer, max_grad_norm: float = 1.0,
                   debug_gradients: bool = True) -> dict[str, float]:
    metrics = gradient_diagnostics(loss, parameters) if debug_gradients else {}
    optimizer.zero_grad(set_to_none=True)
    loss.total_loss.backward()
    before = grad_norm(parameters)
    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    after = grad_norm(parameters)
    if not math.isfinite(before + after):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("non-finite gradient")
    optimizer.step()
    metrics.update({"total_grad_norm_before_clip": before, "total_grad_norm_after_clip": after,
                    "action_policy_loss": float(loss.action_policy_loss.detach()),
                    "psnr_policy_loss": float(loss.psnr_policy_loss.detach()),
                    "weighted_action_policy_loss": float(loss.weighted_action_policy_loss.detach()),
                    "weighted_psnr_policy_loss": float(loss.weighted_psnr_policy_loss.detach()),
                    "raw_kl_loss": float(loss.raw_kl_loss.detach()),
                    "weighted_kl_loss": float(loss.weighted_kl_loss.detach()),
                    "total_loss": float(loss.total_loss.detach())})
    return metrics

