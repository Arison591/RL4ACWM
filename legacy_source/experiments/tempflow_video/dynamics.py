from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class FlowTransition:
    """One reverse SDE transition represented in the GE-Sim EDM coordinate."""

    next_sample: torch.Tensor
    log_prob: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    rf_noise_std: torch.Tensor
    exploration_noise: torch.Tensor
    flow_time: float
    next_flow_time: float


def _validate_times(flow_time: float, next_flow_time: float) -> tuple[float, float, float]:
    t = float(flow_time)
    next_t = float(next_flow_time)
    if not (math.isfinite(t) and math.isfinite(next_t)):
        raise ValueError("flow times must be finite")
    if not 0.0 < t < 1.0:
        raise ValueError(f"flow_time must lie in (0, 1), got {t}")
    if not 0.0 <= next_t < t:
        raise ValueError(
            f"reverse transition requires 0 <= next_flow_time < flow_time, got {next_t} >= {t}"
        )
    return t, next_t, next_t - t


def paper_exploration_scale(flow_time: float, eta: float) -> float:
    """Paper Eq. (5): sigma_t = eta * sqrt(t / (1 - t))."""

    t = float(flow_time)
    eta = float(eta)
    if not 0.0 < t < 1.0:
        raise ValueError(f"flow_time must lie in (0, 1), got {t}")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError(f"eta must be finite and positive, got {eta}")
    return eta * math.sqrt(t / (1.0 - t))


def _normal_log_prob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if sample.shape != mean.shape:
        raise ValueError(f"sample/mean shape mismatch: {sample.shape} != {mean.shape}")
    if torch.any(std <= 0) or not torch.isfinite(std).all():
        raise ValueError("transition std must be finite and positive")
    log_prob = -((sample.detach() - mean) ** 2) / (2.0 * std.square())
    log_prob = log_prob - torch.log(std) - 0.5 * math.log(2.0 * math.pi)
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def edm_transition_mean(
    edm_sample: torch.Tensor,
    velocity: torch.Tensor,
    *,
    flow_time: float,
    next_flow_time: float,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map TempFlow's RF-coordinate Gaussian transition into GE-Sim EDM space.

    GE-Sim stores ``y_t = x_t / (1-t) = x_0 + sigma_edm * epsilon`` while the
    transformer predicts the rectified-flow velocity ``v = epsilon - x_0``.
    TempFlow Eq. (5) is therefore evaluated on ``x_t=(1-t)y_t`` and the mean
    and standard deviation are divided by ``1-next_t`` afterwards.  The
    coordinate Jacobian is independent of policy parameters, so PPO ratios are
    unchanged by this representation change.
    """

    if edm_sample.shape != velocity.shape:
        raise ValueError(f"sample/velocity shape mismatch: {edm_sample.shape} != {velocity.shape}")
    t, next_t, dt = _validate_times(flow_time, next_flow_time)
    exploration = paper_exploration_scale(t, eta)
    rf_sample = (1.0 - t) * edm_sample
    drift = velocity + (exploration**2 / (2.0 * t)) * (
        rf_sample + (1.0 - t) * velocity
    )
    rf_mean = rf_sample + drift * dt
    rf_std = edm_sample.new_tensor(exploration * math.sqrt(-dt))
    edm_mean = rf_mean / (1.0 - next_t)
    edm_std = rf_std / (1.0 - next_t)
    return edm_mean, edm_std, rf_std


def edm_sde_transition_with_logprob(
    edm_sample: torch.Tensor,
    velocity: torch.Tensor,
    *,
    flow_time: float,
    next_flow_time: float,
    eta: float,
    generator: torch.Generator | None = None,
    next_sample: torch.Tensor | None = None,
    exploration_noise: torch.Tensor | None = None,
) -> FlowTransition:
    """Sample or score one TempFlow SDE action in the GE-Sim latent space."""

    if next_sample is not None and exploration_noise is not None:
        raise ValueError("pass either next_sample or exploration_noise, not both")
    mean, std, rf_std = edm_transition_mean(
        edm_sample,
        velocity,
        flow_time=flow_time,
        next_flow_time=next_flow_time,
        eta=eta,
    )
    if next_sample is None:
        if exploration_noise is None:
            exploration_noise = torch.randn(
                edm_sample.shape,
                device=edm_sample.device,
                dtype=edm_sample.dtype,
                generator=generator,
            )
        elif exploration_noise.shape != edm_sample.shape:
            raise ValueError("exploration noise shape must match the latent")
        next_sample = mean + std * exploration_noise
    else:
        if next_sample.shape != edm_sample.shape:
            raise ValueError("next sample shape must match the latent")
        exploration_noise = (next_sample.detach() - mean.detach()) / std
    log_prob = _normal_log_prob(next_sample, mean, std)
    return FlowTransition(
        next_sample=next_sample,
        log_prob=log_prob,
        mean=mean,
        std=std,
        rf_noise_std=rf_std,
        exploration_noise=exploration_noise,
        flow_time=float(flow_time),
        next_flow_time=float(next_flow_time),
    )


def deterministic_edm_step(
    edm_sample: torch.Tensor,
    velocity: torch.Tensor,
    *,
    flow_time: float,
    next_flow_time: float,
) -> torch.Tensor:
    """GE-Sim's deterministic Euler step without mutating scheduler state."""

    t, next_t, _ = _validate_times(flow_time, next_flow_time)
    rf_sample = (1.0 - t) * edm_sample
    predicted_clean = rf_sample - t * velocity
    predicted_noise = rf_sample + (1.0 - t) * velocity
    next_sigma_edm = next_t / max(1.0 - next_t, 1.0e-12)
    return predicted_clean + next_sigma_edm * predicted_noise


def raw_noise_levels(
    flow_times: Sequence[float],
    *,
    eta: float,
) -> torch.Tensor:
    """Return paper noise magnitudes sigma_t*sqrt(-Delta t) for a schedule."""

    if len(flow_times) < 2:
        raise ValueError("at least two flow times are required")
    values = []
    for t, next_t in zip(flow_times[:-1], flow_times[1:]):
        t = float(t)
        next_t = float(next_t)
        if not (0.0 < t < 1.0 and 0.0 <= next_t <= t):
            raise ValueError(f"invalid reverse flow schedule pair: {t} -> {next_t}")
        values.append(
            0.0
            if next_t == t
            else paper_exploration_scale(t, eta) * math.sqrt(t - next_t)
        )
    return torch.tensor(values, dtype=torch.float64)


def noise_aware_weights(
    flow_times: Sequence[float],
    *,
    eta: float,
    enabled: bool = True,
    normalization: str = "schedule_mean",
) -> torch.Tensor:
    """Map paper ``Norm(sigma_t sqrt(Delta t))`` to a concrete schedule.

    The paper leaves ``Norm`` undefined and the official repository uses
    model-specific constants (2.25 for SD3, 1.73 for FLUX, 1.53 for QwenImage).
    For a new video scheduler we avoid inventing another constant: normalize
    the exact paper noise magnitudes by their mean over configured branch
    transitions.  This preserves their ratios and gives mean weight one.
    """

    raw = raw_noise_levels(flow_times, eta=eta)
    if not enabled:
        return torch.ones_like(raw)
    if normalization == "none":
        return raw
    if normalization != "schedule_mean":
        raise ValueError(f"unknown noise-aware normalization: {normalization}")
    # GE-Sim appends a duplicate terminal sigma for ``final_sigmas_type=sigma_min``.
    # That pair is a scheduler no-op and cannot be sampled as an SDE action, so it
    # must not dilute the normalization over genuinely branchable transitions.
    branchable = raw[raw > 0]
    denominator = branchable.mean() if branchable.numel() else raw.new_tensor(float("nan"))
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("noise-aware normalization denominator must be positive")
    return raw / denominator
