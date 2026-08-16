from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class FlowTransition:
    next_sample: torch.Tensor
    log_prob: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    rf_noise_std: torch.Tensor
    exploration_noise: torch.Tensor
    flow_time: float
    next_flow_time: float


def _times(flow_time: float, next_flow_time: float) -> tuple[float, float, float]:
    t, nxt = float(flow_time), float(next_flow_time)
    if not (math.isfinite(t) and math.isfinite(nxt) and 0.0 < t < 1.0 and 0.0 <= nxt < t):
        raise ValueError(f"invalid reverse flow transition {t} -> {nxt}")
    return t, nxt, nxt - t


def paper_exploration_scale(flow_time: float, eta: float) -> float:
    t, eta = float(flow_time), float(eta)
    if not 0.0 < t < 1.0 or not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("flow_time must be in (0,1) and eta must be positive")
    return eta * math.sqrt(t / (1.0 - t))


def _normal_log_prob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if sample.shape != mean.shape or torch.any(std <= 0) or not torch.isfinite(std).all():
        raise ValueError("invalid Gaussian transition")
    value = -((sample.detach() - mean) ** 2) / (2.0 * std.square())
    value = value - torch.log(std) - 0.5 * math.log(2.0 * math.pi)
    return value.mean(dim=tuple(range(1, value.ndim)))


def edm_transition_mean(edm_sample: torch.Tensor, velocity: torch.Tensor, *, flow_time: float,
                        next_flow_time: float, eta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if edm_sample.shape != velocity.shape:
        raise ValueError("sample/velocity shape mismatch")
    t, nxt, dt = _times(flow_time, next_flow_time)
    exploration = paper_exploration_scale(t, eta)
    rf_sample = (1.0 - t) * edm_sample
    drift = velocity + (exploration**2 / (2.0 * t)) * (rf_sample + (1.0 - t) * velocity)
    rf_mean = rf_sample + drift * dt
    rf_std = edm_sample.new_tensor(exploration * math.sqrt(-dt))
    return rf_mean / (1.0 - nxt), rf_std / (1.0 - nxt), rf_std


def edm_sde_transition_with_logprob(edm_sample: torch.Tensor, velocity: torch.Tensor, *,
                                    flow_time: float, next_flow_time: float, eta: float,
                                    generator: torch.Generator | None = None,
                                    next_sample: torch.Tensor | None = None,
                                    exploration_noise: torch.Tensor | None = None) -> FlowTransition:
    if next_sample is not None and exploration_noise is not None:
        raise ValueError("pass next_sample or exploration_noise, not both")
    mean, std, rf_std = edm_transition_mean(edm_sample, velocity, flow_time=flow_time,
                                             next_flow_time=next_flow_time, eta=eta)
    if next_sample is None:
        if exploration_noise is None:
            exploration_noise = torch.randn(edm_sample.shape, device=edm_sample.device,
                                              dtype=edm_sample.dtype, generator=generator)
        if exploration_noise.shape != edm_sample.shape:
            raise ValueError("exploration noise shape mismatch")
        next_sample = mean + std * exploration_noise
    else:
        if next_sample.shape != edm_sample.shape:
            raise ValueError("next sample shape mismatch")
        exploration_noise = (next_sample.detach() - mean.detach()) / std
    return FlowTransition(next_sample, _normal_log_prob(next_sample, mean, std), mean, std,
                          rf_std, exploration_noise, float(flow_time), float(next_flow_time))


def deterministic_edm_step(edm_sample: torch.Tensor, velocity: torch.Tensor, *,
                           flow_time: float, next_flow_time: float) -> torch.Tensor:
    t, nxt, _ = _times(flow_time, next_flow_time)
    rf_sample = (1.0 - t) * edm_sample
    clean = rf_sample - t * velocity
    noise = rf_sample + (1.0 - t) * velocity
    return clean + (nxt / max(1.0 - nxt, 1.0e-12)) * noise


def raw_noise_levels(flow_times: Sequence[float], *, eta: float) -> torch.Tensor:
    if len(flow_times) < 2:
        raise ValueError("at least two flow times required")
    values = []
    for t, nxt in zip(flow_times[:-1], flow_times[1:]):
        if not (0.0 < float(t) < 1.0 and 0.0 <= float(nxt) <= float(t)):
            raise ValueError("invalid schedule")
        values.append(0.0 if nxt == t else paper_exploration_scale(t, eta) * math.sqrt(t - nxt))
    return torch.tensor(values, dtype=torch.float64)


def noise_aware_weights(flow_times: Sequence[float], *, eta: float, enabled: bool = True,
                        normalization: str = "schedule_mean") -> torch.Tensor:
    raw = raw_noise_levels(flow_times, eta=eta)
    if not enabled:
        return torch.ones_like(raw)
    if normalization == "none":
        return raw
    if normalization != "schedule_mean":
        raise ValueError(f"unknown normalization {normalization}")
    branchable = raw[raw > 0]
    denominator = branchable.mean() if branchable.numel() else raw.new_tensor(float("nan"))
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("invalid noise normalization denominator")
    return raw / denominator

