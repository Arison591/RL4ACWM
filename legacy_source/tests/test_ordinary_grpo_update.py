from pathlib import Path

from experiments.tempflow_video.policy import ReferencePolicyAdapter
from experiments.tempflow_video.schemas import (
    CollectedTransition,
    OrdinaryGroupKey,
    OrdinaryRollout,
)
from experiments.tempflow_video.trainer import TempFlowOptimizerConfig, TempFlowVideoTrainer
from tests.tempflow_test_utils import ToyVideoPolicy, make_toy_rollouts


def test_ordinary_multistep_grpo_updates_policy(tmp_path: Path):
    policy = ToyVideoPolicy()
    source = make_toy_rollouts(policy, tmp_path)
    key = OrdinaryGroupKey("condition", "prompt", 29, "reward-hash", 0)
    rollouts = []
    for index, item in enumerate(source):
        action = CollectedTransition(
            timestep=2,
            current_latent=item.current_latent,
            next_latent=item.next_latent,
            flow_time=item.flow_time,
            next_flow_time=item.next_flow_time,
            eta=item.eta,
            old_log_prob=item.old_log_prob,
            rf_noise_std=item.rf_noise_std,
            noise_weight=item.noise_weight,
            noise_seed=item.branch_noise_seed,
        )
        rollouts.append(
            OrdinaryRollout(
                sample_id=f"ordinary-{index}",
                group_key=key,
                policy_version=0,
                rollout_id=index,
                initial_seed=100 + index,
                seed_dir=tmp_path / f"ordinary-{index}",
                condition=None,
                transitions=[action, action],
                reward=item.reward,
                advantage=item.advantage,
            )
        )
    reference = ReferencePolicyAdapter(policy)
    trainer = TempFlowVideoTrainer(
        policy,
        reference,
        TempFlowOptimizerConfig(learning_rate=0.05, clip_range=0.2, log_term_grad_norm=False),
    )
    before = policy.policy_model.delta.detach().clone()
    record = trainer.update_group(rollouts)
    assert record.optimizer_step == 1
    assert policy.policy_model.delta.detach().ne(before)
    reference.assert_unchanged()
