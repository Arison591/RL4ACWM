import pytest
import torch

from experiments.tempflow_video.dynamics import (
    deterministic_edm_step,
    edm_sde_transition_with_logprob,
    edm_transition_mean,
    noise_aware_weights,
)


def test_branches_share_state_but_use_distinct_noise():
    current = torch.arange(12, dtype=torch.float64).reshape(1, 3, 2, 2)
    velocity = torch.full_like(current, 0.125)
    first = edm_sde_transition_with_logprob(
        current,
        velocity,
        flow_time=0.8,
        next_flow_time=0.7,
        eta=0.7,
        generator=torch.Generator().manual_seed(11),
    )
    second = edm_sde_transition_with_logprob(
        current.clone(),
        velocity,
        flow_time=0.8,
        next_flow_time=0.7,
        eta=0.7,
        generator=torch.Generator().manual_seed(12),
    )

    assert torch.equal(current, current.clone())
    assert torch.equal(first.mean, second.mean)
    assert not torch.equal(first.exploration_noise, second.exploration_noise)
    assert not torch.equal(first.next_sample, second.next_sample)
    assert first.log_prob.shape == (1,)
    assert torch.isfinite(first.log_prob).all()


def test_collected_transition_recomputes_exact_logprob_and_has_gradient():
    current = torch.randn(2, 3, 4, dtype=torch.float64)
    old_velocity = torch.randn_like(current)
    collected = edm_sde_transition_with_logprob(
        current,
        old_velocity,
        flow_time=0.65,
        next_flow_time=0.5,
        eta=0.7,
        generator=torch.Generator().manual_seed(5),
    )
    train_velocity = old_velocity.clone().requires_grad_(True)
    rescored = edm_sde_transition_with_logprob(
        current,
        train_velocity,
        flow_time=0.65,
        next_flow_time=0.5,
        eta=0.7,
        next_sample=collected.next_sample,
    )

    assert torch.allclose(rescored.log_prob, collected.log_prob, atol=1.0e-12, rtol=0.0)
    (-rescored.log_prob.mean()).backward()
    assert train_velocity.grad is not None
    assert torch.isfinite(train_velocity.grad).all()
    assert train_velocity.grad.norm().item() > 0.0


def test_edm_mean_is_coordinate_transform_of_paper_rf_mean():
    y = torch.tensor([[[2.0, -1.0]]], dtype=torch.float64)
    v = torch.tensor([[[0.3, 0.4]]], dtype=torch.float64)
    t, next_t, eta = 0.6, 0.45, 0.7
    mean_y, std_y, std_rf = edm_transition_mean(
        y, v, flow_time=t, next_flow_time=next_t, eta=eta
    )

    x = (1.0 - t) * y
    sigma = eta * (t / (1.0 - t)) ** 0.5
    dt = next_t - t
    expected_x = x + (v + sigma**2 / (2.0 * t) * (x + (1.0 - t) * v)) * dt
    assert torch.allclose(mean_y, expected_x / (1.0 - next_t))
    assert torch.allclose(std_y, std_rf / (1.0 - next_t))


def test_deterministic_edm_step_matches_clean_noise_reconstruction():
    clean = torch.randn(2, 3, dtype=torch.float64)
    noise = torch.randn_like(clean)
    t, next_t = 0.75, 0.4
    sigma = t / (1.0 - t)
    next_sigma = next_t / (1.0 - next_t)
    current = clean + sigma * noise
    velocity = noise - clean

    actual = deterministic_edm_step(
        current, velocity, flow_time=t, next_flow_time=next_t
    )

    assert torch.allclose(actual, clean + next_sigma * noise, atol=1.0e-12)


def test_noise_aware_weights_derive_from_noise_schedule_not_step_index():
    times = [0.9, 0.7, 0.4, 0.1]
    weights = noise_aware_weights(times, eta=0.7)

    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[0] > weights[1] > weights[2]
    assert torch.equal(
        noise_aware_weights(times, eta=0.7, enabled=False),
        torch.ones(3, dtype=torch.float64),
    )
