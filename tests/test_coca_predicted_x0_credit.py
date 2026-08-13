from __future__ import annotations

import pytest
import torch

from experiments.awm_coca.coca_credit import compute_credit


def _trajectory(*, include_denoised: bool = True) -> dict:
    rows = [
        {"step": 0, "timestep": 1.0, "latents": torch.tensor([[9.0, 9.0]])},
        {"step": 1, "timestep": 0.8, "latents": torch.tensor([[1.0, 0.0]])},
        # This is also the final post-step latent used as CoCA's endpoint.
        {"step": 2, "timestep": 0.0, "latents": torch.tensor([[1.0, 0.0]])},
    ]
    if include_denoised:
        rows[1]["denoised"] = torch.tensor([[0.0, 1.0]])
        rows[2]["denoised"] = torch.tensor([[1.0, 0.0]])
    return {"chunks": [rows], "num_chunks": 1}


def test_predicted_x0_credit_compares_denoised_predictions_to_final_latent():
    credit = compute_credit(
        _trajectory(),
        0.75,
        window_size=1,
        num_training_noise_levels=2,
    )

    assert credit["credit_source"] == "predicted_x0"
    assert credit["num_reverse_steps"] == 2
    assert [row["cosine_similarity"] for row in credit["step_rows"]] == pytest.approx([0.0, 1.0])
    # all_x0=[x0_hat_1, x0_hat_2, final], hence deltas=[1, 0].
    assert [row["delta_similarity"] for row in credit["step_rows"]] == pytest.approx([1.0, 0.0])
    assert credit["reward_conservation_error"] == pytest.approx(0.0)


def test_predicted_x0_credit_rejects_legacy_trajectory_without_denoised():
    with pytest.raises(ValueError, match="no predicted-x0"):
        compute_credit(_trajectory(include_denoised=False), 1.0)
