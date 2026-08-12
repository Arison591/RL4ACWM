from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ProposalConfig:
    noise_levels: tuple[float, ...]
    eta: float = 0.5
    temperature: float = 1.0
    base_probabilities: tuple[float, ...] | None = None
    importance_clipping: float | None = None

    def __post_init__(self) -> None:
        if not self.noise_levels:
            raise ValueError("noise_levels must not be empty")
        if not 0.0 <= self.eta < 1.0:
            raise ValueError("eta must satisfy 0 <= eta < 1")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if any(not 0.0 <= value <= 1.0 for value in self.noise_levels):
            raise ValueError("noise_levels must lie in [0, 1]")
        if self.base_probabilities is not None:
            if len(self.base_probabilities) != len(self.noise_levels):
                raise ValueError("base_probabilities and noise_levels must have equal length")
            if any(value <= 0.0 for value in self.base_probabilities):
                raise ValueError("base_probabilities must have full support")
        if self.importance_clipping is not None and self.importance_clipping <= 0.0:
            raise ValueError("importance_clipping must be positive when configured")


@dataclass(frozen=True)
class RolloutTrainingSample:
    sample_id: str
    condition_id: str
    policy_version: int
    clean_latent: torch.Tensor
    condition: Any
    advantage: float
    reward: float
    noise_scores: Sequence[float]


@dataclass
class LossRecord:
    sample_id: str
    noise_level_index: int
    noise_time: float
    base_probability: float
    proposal_probability: float
    importance_weight: float
    advantage: float
    fm_loss: float
    reference_kl: float
    weighted_loss: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VelocityModelAdapter(Protocol):
    def policy_velocity(
        self, noisy_latent: torch.Tensor, noise_time: torch.Tensor, condition: Any
    ) -> torch.Tensor: ...

    def reference_velocity(
        self, noisy_latent: torch.Tensor, noise_time: torch.Tensor, condition: Any
    ) -> torch.Tensor: ...


def leave_one_out_advantages(rewards: Sequence[float]) -> tuple[list[float], list[float]]:
    if len(rewards) < 2:
        raise ValueError("leave-one-out advantage requires at least two rollouts")
    total = float(sum(rewards))
    denominator = len(rewards) - 1
    baselines = [(total - float(reward)) / denominator for reward in rewards]
    return baselines, [float(reward) - baseline for reward, baseline in zip(rewards, baselines)]


def base_distribution(config: ProposalConfig, *, device: torch.device) -> torch.Tensor:
    if config.base_probabilities is None:
        return torch.full((len(config.noise_levels),), 1.0 / len(config.noise_levels), device=device)
    probabilities = torch.as_tensor(config.base_probabilities, dtype=torch.float32, device=device)
    return probabilities / probabilities.sum()


def build_proposal(
    noise_scores: Sequence[float] | torch.Tensor,
    config: ProposalConfig,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.as_tensor(noise_scores, dtype=torch.float32, device=device).detach()
    if scores.numel() != len(config.noise_levels):
        raise ValueError("one CoCA score is required for each training noise level")
    if not torch.isfinite(scores).all():
        raise ValueError("noise_scores must be finite")
    base = base_distribution(config, device=device)
    coca = torch.softmax(scores / config.temperature, dim=0)
    proposal = (1.0 - config.eta) * base + config.eta * coca
    return base, coca, proposal


def forward_noise(clean_latent: torch.Tensor, noise: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    expand = (clean_latent.ndim - time.ndim) * (1,)
    shaped_time = time.reshape(*time.shape, *expand)
    return (1.0 - shaped_time) * clean_latent + shaped_time * noise


def awm_coca_is_loss(
    adapter: VelocityModelAdapter,
    samples: Sequence[RolloutTrainingSample],
    proposal_config: ProposalConfig,
    *,
    beta: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[LossRecord]]:
    if not samples:
        raise ValueError("samples must not be empty")
    losses: list[torch.Tensor] = []
    records: list[LossRecord] = []
    for sample in samples:
        sample_loss, record = awm_coca_sample_loss(
            adapter, sample, proposal_config, beta=beta, generator=generator
        )
        losses.append(sample_loss)
        records.append(record)
    return torch.stack(losses).mean(), records


def awm_coca_sample_loss(
    adapter: VelocityModelAdapter,
    sample: RolloutTrainingSample,
    proposal_config: ProposalConfig,
    *,
    beta: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, LossRecord]:
    """Build one rollout loss so a trainer can backward without retaining a whole group graph."""
    clean = sample.clean_latent.detach()
    base, _, proposal = build_proposal(sample.noise_scores, proposal_config, device=clean.device)
    level = int(torch.multinomial(proposal, 1, generator=generator).item())
    probability = proposal[level].detach()
    importance = (base[level] / probability).detach()
    if proposal_config.importance_clipping is not None:
        importance = importance.clamp(max=proposal_config.importance_clipping)
    time = torch.tensor([proposal_config.noise_levels[level]], device=clean.device, dtype=clean.dtype)
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    noisy = forward_noise(clean, noise, time)
    target = noise - clean
    prediction = adapter.policy_velocity(noisy, time, sample.condition)
    with torch.no_grad():
        reference = adapter.reference_velocity(noisy, time, sample.condition)
    fm_loss = F.mse_loss(prediction.float(), target.float())
    reference_kl = F.mse_loss(prediction.float(), reference.float())
    sample_loss = importance * (float(sample.advantage) * fm_loss + float(beta) * reference_kl)
    return sample_loss, LossRecord(
        sample_id=sample.sample_id,
        noise_level_index=level,
        noise_time=float(time.item()),
        base_probability=float(base[level].item()),
        proposal_probability=float(probability.item()),
        importance_weight=float(importance.item()),
        advantage=float(sample.advantage),
        fm_loss=float(fm_loss.detach().item()),
        reference_kl=float(reference_kl.detach().item()),
        weighted_loss=float(sample_loss.detach().item()),
    )
