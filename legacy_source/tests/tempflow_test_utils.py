from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.tempflow_video.dynamics import edm_sde_transition_with_logprob
from experiments.tempflow_video.schemas import BranchGroupKey, BranchRollout


class ToyVideoPolicy:
    def __init__(self):
        self.policy_model = torch.nn.Module()
        self.policy_model.delta = torch.nn.Parameter(torch.tensor(0.0))
        self.policy_model.base = torch.nn.Parameter(torch.tensor(0.2), requires_grad=False)
        self.runtime = SimpleNamespace(device=torch.device("cpu"))

    def predict_velocity_or_noise(self, latent, flow_time, condition, *, reference=False):
        base = self.policy_model.base * torch.ones_like(latent)
        return base if reference else base + self.policy_model.delta * torch.ones_like(latent)

    def get_reference_prediction(self, latent, flow_time, condition):
        return self.predict_velocity_or_noise(latent, flow_time, condition, reference=True)

    def get_trainable_parameters(self):
        return [self.policy_model.delta]

    def reference_parameters(self):
        yield "base", self.policy_model.base


def make_toy_rollouts(policy, tmp_path: Path):
    key = BranchGroupKey(
        condition_id="condition",
        prompt_id="prompt",
        initial_seed=123,
        branch_timestep=2,
        video_length=29,
        reward_config_sha256="reward-hash",
    )
    current = torch.tensor([[0.1, -0.2, 0.3, 0.5]], dtype=torch.float32)
    velocity = policy.predict_velocity_or_noise(current, 0.7, None)
    rollouts = []
    for branch_id, seed in enumerate((11, 12)):
        transition = edm_sde_transition_with_logprob(
            current,
            velocity,
            flow_time=0.7,
            next_flow_time=0.5,
            eta=0.7,
            generator=torch.Generator().manual_seed(seed),
        )
        reward = 1.0 if branch_id == 0 else 0.0
        rollout = BranchRollout(
            sample_id=f"sample-{branch_id}",
            group_key=key,
            policy_version=0,
            branch_id=branch_id,
            branch_noise_seed=seed,
            seed_dir=tmp_path / f"branch-{branch_id}",
            current_latent=current.clone(),
            next_latent=transition.next_sample.detach().clone(),
            condition=None,
            flow_time=0.7,
            next_flow_time=0.5,
            eta=0.7,
            old_log_prob=float(transition.log_prob.mean().item()),
            rf_noise_std=float(transition.rf_noise_std.item()),
            noise_weight=1.0,
            prefix_latent_sha256="same-prefix",
            branch_noise_sha256=f"noise-{branch_id}",
            reward={"total_reward": reward, "valid": True},
            advantage=1.0 if branch_id == 0 else -1.0,
        )
        rollouts.append(rollout)
    return rollouts
