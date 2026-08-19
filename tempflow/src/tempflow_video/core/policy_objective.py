from __future__ import annotations

from dataclasses import dataclass

import torch

from .reference_kl import closed_form_equal_variance_kl


@dataclass(frozen=True)
class PolicyLoss:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    raw_kl_loss: torch.Tensor
    weighted_kl_loss: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_mean: torch.Tensor


@dataclass(frozen=True)
class ComponentPolicyLoss:
    total_loss: torch.Tensor
    action_policy_loss: torch.Tensor
    psnr_policy_loss: torch.Tensor
    weighted_action_policy_loss: torch.Tensor
    weighted_psnr_policy_loss: torch.Tensor
    raw_kl_loss: torch.Tensor
    weighted_kl_loss: torch.Tensor


def grpo_surrogate(log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                   advantages: torch.Tensor, noise_weights: torch.Tensor,
                   clip_range: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    loss = -(noise_weights * torch.minimum(ratio * advantages, clipped * advantages)).mean()
    return loss, 0.5 * log_ratio.square().mean(), (torch.abs(ratio - 1) > clip_range).float().mean(), ratio.mean()


def legacy_policy_objective(*, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                            advantages: torch.Tensor, noise_weights: torch.Tensor,
                            policy_means: torch.Tensor, reference_means: torch.Tensor,
                            transition_stds: torch.Tensor, clip_range: float,
                            kl_beta: float) -> PolicyLoss:
    policy, approx, fraction, ratio = grpo_surrogate(log_probs, old_log_probs, advantages,
                                                      noise_weights, clip_range)
    raw_kl = closed_form_equal_variance_kl(policy_means, reference_means, transition_stds)
    weighted_kl = kl_beta * raw_kl
    return PolicyLoss(policy + weighted_kl, policy, raw_kl, weighted_kl, approx, fraction, ratio)


def component_policy_objective(*, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                               action_advantages: torch.Tensor, psnr_advantages: torch.Tensor,
                               noise_weights: torch.Tensor, policy_means: torch.Tensor,
                               reference_means: torch.Tensor, transition_stds: torch.Tensor,
                               clip_range: float, lambda_action: float = 0.5,
                               lambda_psnr: float = 0.5, kl_beta: float = 0.01) -> ComponentPolicyLoss:
    action, _, _, _ = grpo_surrogate(log_probs, old_log_probs, action_advantages, noise_weights, clip_range)
    psnr, _, _, _ = grpo_surrogate(log_probs, old_log_probs, psnr_advantages, noise_weights, clip_range)
    raw_kl = closed_form_equal_variance_kl(policy_means, reference_means, transition_stds)
    wa, wp, wk = lambda_action * action, lambda_psnr * psnr, kl_beta * raw_kl
    return ComponentPolicyLoss(wa + wp + wk, action, psnr, wa, wp, raw_kl, wk)


def gradient_diagnostics(loss: ComponentPolicyLoss, parameters: list[torch.nn.Parameter]) -> dict[str, float]:
    names = ("action_policy", "psnr_policy", "weighted_action", "weighted_psnr", "weighted_kl")
    terms = (loss.action_policy_loss, loss.psnr_policy_loss, loss.weighted_action_policy_loss,
             loss.weighted_psnr_policy_loss, loss.weighted_kl_loss)
    vectors = {}
    for name, term in zip(names, terms):
        grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
        vectors[name] = torch.cat([g.detach().reshape(-1) for g in grads if g is not None])
    def norm(v): return float(torch.linalg.vector_norm(v)) if v.numel() else 0.0
    a, p = vectors["action_policy"], vectors["psnr_policy"]
    cosine = float(torch.nn.functional.cosine_similarity(a, p, dim=0)) if a.numel() and p.numel() else 0.0
    return {"action_policy_grad_norm": norm(a), "psnr_policy_grad_norm": norm(p),
            "weighted_action_grad_norm": norm(vectors["weighted_action"]),
            "weighted_psnr_grad_norm": norm(vectors["weighted_psnr"]),
            "weighted_kl_grad_norm": norm(vectors["weighted_kl"]),
            "action_psnr_grad_cosine": cosine}
