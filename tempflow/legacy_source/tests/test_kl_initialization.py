import pytest
import torch

from experiments.tempflow_video.dynamics import edm_transition_mean
from experiments.tempflow_video.policy import ReferencePolicyAdapter
from tests.tempflow_test_utils import ToyVideoPolicy


def test_policy_and_reference_match_at_initialization_with_zero_kl():
    policy = ToyVideoPolicy()
    reference = ReferencePolicyAdapter(policy)
    latent = torch.randn(2, 4)
    policy_velocity = policy.predict_velocity_or_noise(latent, 0.7, None)
    reference_velocity = reference.predict_velocity_or_noise(latent, 0.7, None)
    policy_mean, std, _ = edm_transition_mean(
        latent, policy_velocity, flow_time=0.7, next_flow_time=0.5, eta=0.7
    )
    reference_mean, _, _ = edm_transition_mean(
        latent, reference_velocity, flow_time=0.7, next_flow_time=0.5, eta=0.7
    )
    kl = (policy_mean - reference_mean).square().mean() / (2.0 * std.square())

    assert torch.equal(policy_velocity, reference_velocity)
    assert kl.item() == pytest.approx(0.0, abs=0.0)
