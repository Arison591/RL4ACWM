import pytest
import torch

from experiments.tempflow_video.dynamics import edm_transition_mean
from experiments.tempflow_video.policy import ReferencePolicyAdapter
from experiments.tempflow_video.trainer import TempFlowOptimizerConfig, TempFlowVideoTrainer
from tests.tempflow_test_utils import ToyVideoPolicy, make_toy_rollouts


def test_real_tempflow_optimizer_step_changes_only_policy(tmp_path):
    policy = ToyVideoPolicy()
    reference = ReferencePolicyAdapter(policy)
    trainer = TempFlowVideoTrainer(
        policy,
        reference,
        TempFlowOptimizerConfig(
            learning_rate=0.05,
            warmup_steps=0,
            clip_range=0.2,
            kl_beta=0.01,
            log_term_grad_norm=True,
        ),
    )
    rollouts = make_toy_rollouts(policy, tmp_path)
    initial_delta = policy.policy_model.delta.detach().clone()
    initial_base = policy.policy_model.base.detach().clone()

    record = trainer.update_group(rollouts)

    assert record.optimizer_step == 1
    assert record.policy_version == 1
    assert record.metrics["policy_grad_norm"] > 0.0
    assert record.metrics["total_grad_norm_before_clip"] > 0.0
    assert record.metrics["raw_kl_loss"] == pytest.approx(0.0, abs=1.0e-12)
    assert record.metrics["parameter_delta_norm"] > 0.0
    assert record.metrics["changed_trainable_parameter_tensors"] == 1.0
    assert torch.not_equal(policy.policy_model.delta.detach(), initial_delta)
    assert torch.equal(policy.policy_model.base.detach(), initial_base)
    reference.assert_unchanged()

    latent = rollouts[0].current_latent
    policy_mean, std, _ = edm_transition_mean(
        latent,
        policy.predict_velocity_or_noise(latent, 0.7, None),
        flow_time=0.7,
        next_flow_time=0.5,
        eta=0.7,
    )
    reference_mean, _, _ = edm_transition_mean(
        latent,
        reference.predict_velocity_or_noise(latent, 0.7, None),
        flow_time=0.7,
        next_flow_time=0.5,
        eta=0.7,
    )
    post_kl = (policy_mean - reference_mean).square().mean() / (2.0 * std.square())
    assert post_kl.item() > 0.0
