from types import MethodType

import torch

from experiments.awm_coca.gesim_runtime import PersistentGeSimRuntime, PreparedGeSimCondition


def test_rollout_group_forks_hidden_global_rng(tmp_path):
    runtime = PersistentGeSimRuntime.__new__(PersistentGeSimRuntime)
    runtime.config = {"rollout": {"group_size": 1}}
    runtime.device = torch.device("cpu")
    runtime.policy_version = 0
    runtime.transformer = torch.nn.Linear(1, 1)

    def fake_batch(self, prepared, *, seeds, group_dir, prompt):
        return [float(torch.rand(()).item())]

    runtime._rollout_batch = MethodType(fake_batch, runtime)
    prepared = PreparedGeSimCondition(
        condition_id="condition",
        sample_dir="unused",
        observation=torch.empty(0),
        cond_to_concat=torch.empty(0),
        original_trajectory=torch.empty(0),
    )
    torch.manual_seed(999)
    expected_after = torch.rand(())
    torch.manual_seed(999)
    _, first = runtime.rollout_group(
        prepared, seeds=[123], output_dir=tmp_path / "a", expected_group_size=1
    )
    after = torch.rand(())
    _, second = runtime.rollout_group(
        prepared, seeds=[123], output_dir=tmp_path / "b", expected_group_size=1
    )

    assert first == second
    assert torch.equal(after, expected_after)
