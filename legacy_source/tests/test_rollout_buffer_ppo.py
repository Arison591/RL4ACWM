from dataclasses import replace

import pytest

from experiments.tempflow_video.policy import ReferencePolicyAdapter
from experiments.tempflow_video.trainer import TempFlowOptimizerConfig, TempFlowVideoTrainer
from tests.tempflow_test_utils import ToyVideoPolicy, make_toy_rollouts


def test_frozen_old_policy_buffer_makes_later_ratio_nontrivial(tmp_path):
    policy = ToyVideoPolicy()
    reference = ReferencePolicyAdapter(policy)
    trainer = TempFlowVideoTrainer(
        policy,
        reference,
        TempFlowOptimizerConfig(
            learning_rate=0.05,
            clip_range=1.0e-4,
            kl_beta=0.0,
            log_term_grad_norm=False,
            log_gradient_cosine=True,
        ),
    )
    first = make_toy_rollouts(policy, tmp_path / "first")
    second = make_toy_rollouts(policy, tmp_path / "second")
    second_key = replace(second[0].group_key, branch_timestep=3)
    for index, rollout in enumerate(second):
        rollout.group_key = second_key
        rollout.sample_id = f"second-{index}"

    records = trainer.update_rollout_buffer(
        [first, second],
        minibatches_per_epoch=2,
        num_inner_epochs=1,
        shuffle=False,
    )

    assert len(records) == 2
    assert records[0].metrics["ratio_mean"] == pytest.approx(1.0)
    assert records[0].metrics["clip_fraction"] == pytest.approx(0.0)
    assert records[1].metrics["ratio_mean"] != pytest.approx(1.0)
    assert records[1].metrics["clip_fraction"] > 0.0
    assert records[0].metrics["gradient_cosine_has_previous_step"] == 0.0
    assert records[1].metrics["gradient_cosine_has_previous_step"] == 1.0
    assert -1.0 <= records[1].metrics["gradient_cosine_with_previous_step"] <= 1.0

    with pytest.raises(ValueError, match="already consumed"):
        trainer.update_rollout_buffer([first])


def test_one_timestep_group_reused_for_two_inner_epochs_activates_ratio(tmp_path):
    policy = ToyVideoPolicy()
    trainer = TempFlowVideoTrainer(
        policy,
        ReferencePolicyAdapter(policy),
        TempFlowOptimizerConfig(
            learning_rate=0.05,
            clip_range=1.0e-4,
            kl_beta=0.0,
            log_term_grad_norm=False,
        ),
    )
    group = make_toy_rollouts(policy, tmp_path / "single")

    records = trainer.update_rollout_buffer(
        [group],
        minibatches_per_epoch=1,
        num_inner_epochs=2,
        shuffle=False,
    )

    assert len(records) == 2
    assert records[0].metrics["ratio_mean"] == pytest.approx(1.0)
    assert records[1].metrics["ratio_mean"] != pytest.approx(1.0)
    assert records[1].metrics["clip_fraction"] > 0.0
