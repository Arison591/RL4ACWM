from __future__ import annotations

import pytest
import torch

from experiments.awm_coca.training_core import ProposalConfig, build_proposal


def test_negative_window_scores_still_produce_valid_sampling_distribution():
    config = ProposalConfig(
        noise_levels=(0.1, 0.2, 0.3),
        eta=0.9,
        temperature=1.0,
        base_probabilities=(1 / 3, 1 / 3, 1 / 3),
    )

    base, coca, proposal = build_proposal(
        [-20.0, 6.0, 15.0], config, device=torch.device("cpu")
    )

    assert base.sum().item() == pytest.approx(1.0)
    assert coca.sum().item() == pytest.approx(1.0)
    assert proposal.sum().item() == pytest.approx(1.0)
    assert torch.all(coca >= 0)
    assert torch.all(proposal > 0)
    # This is the operation used by the real training path and used to fail on
    # negative raw CoCA scores.
    assert 0 <= int(torch.multinomial(proposal, 1).item()) < 3


def test_temperature_controls_coca_sharpness():
    scores = [0.0, 1.0]
    cold = ProposalConfig(noise_levels=(0.2, 0.8), eta=0.5, temperature=0.1)
    warm = ProposalConfig(noise_levels=(0.2, 0.8), eta=0.5, temperature=10.0)

    _, cold_coca, _ = build_proposal(scores, cold, device=torch.device("cpu"))
    _, warm_coca, _ = build_proposal(scores, warm, device=torch.device("cpu"))

    assert cold_coca[1] - cold_coca[0] > warm_coca[1] - warm_coca[0]
