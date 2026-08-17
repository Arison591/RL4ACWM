from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

import torch
import torch.nn.functional as F

from experiments.awm_coca.advantage import leave_one_out_advantages


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
    action_advantage: float | None = None
    geometry_advantage: float | None = None


@dataclass
class LossRecord:
    sample_id: str
    noise_level_index: int
    noise_time: float
    base_probability: float
    proposal_probability: float
    importance_weight: float
    advantage: float
    action_advantage: float | None
    geometry_advantage: float | None
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
    # CoCA window contributions are scores, not probabilities: a non-monotonic
    # similarity trajectory can legitimately produce negative values.  The
    # temperature softmax matches haoran/CoCA and guarantees valid sampling
    # probabilities before mixing with the full-support base distribution.
    coca = torch.softmax(scores / float(config.temperature), dim=0)
    proposal = (1.0 - config.eta) * base + config.eta * coca
    if not torch.isfinite(proposal).all() or torch.any(proposal <= 0):
        raise ValueError("CoCA proposal must be finite and strictly positive")
    proposal = proposal / proposal.sum()
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
        sample_loss, record, _, _ = awm_coca_sample_loss(
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
) -> tuple[torch.Tensor, LossRecord, torch.Tensor, torch.Tensor]:
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
    # GE-Sim sampler 使用 EDM 式未缩放 latent：x_sigma = x0 + sigma·noise，sigma = t/(1-t)。
    # adapter 喂模型前做 c_in=(1-t) 缩放，恰好得到 rectified-flow 输入 (1-t)x0 + t·noise，
    # 对应 transformer 原生 flow-matching 输出 target = noise - clean（与部署 pipeline
    # 的速度约定一致：pipeline 在 x0 + sigma·noise 路径上以 (x - denoised)/sigma 积分）。
    sigma = time / (1.0 - time).clamp_min(1e-6)
    shaped_sigma = sigma.reshape(*sigma.shape, *((1,) * (clean.ndim - sigma.ndim)))
    noisy = clean + shaped_sigma * noise
    target = noise - clean
    prediction = adapter.policy_velocity(noisy, time, sample.condition)
    with torch.no_grad():
        reference = adapter.reference_velocity(noisy, time, sample.condition)
    fm_loss = F.mse_loss(prediction.float(), target.float())
    reference_kl = F.mse_loss(prediction.float(), reference.float())
    # 拆成两项返回：fm 项与 reference-KL 项，trainer 据此逐项统计梯度范数
    # （importance 是两项的公共因子，拆后不影响相对梯度尺度）。
    advantage = float(sample.advantage)
    fm_term = importance * (advantage * fm_loss)
    kl_term = importance * (float(beta) * reference_kl)
    sample_loss = fm_term + kl_term
    return sample_loss, LossRecord(
        sample_id=sample.sample_id,
        noise_level_index=level,
        noise_time=float(time.item()),
        base_probability=float(base[level].item()),
        proposal_probability=float(probability.item()),
        importance_weight=float(importance.item()),
        advantage=advantage,
        action_advantage=sample.action_advantage,
        geometry_advantage=sample.geometry_advantage,
        fm_loss=float(fm_loss.detach().item()),
        reference_kl=float(reference_kl.detach().item()),
        weighted_loss=float(sample_loss.detach().item()),
    ), fm_term, kl_term
