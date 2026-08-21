from types import SimpleNamespace

import torch

from experiments.tempflow_video.distributed import partition_indices, weighted_mean_dict
from experiments.tempflow_video.sampler import VideoTrajectorySampler


def test_six_branches_are_sharded_across_four_ranks_without_drop():
    shards = [partition_indices(6, 4, rank) for rank in range(4)]
    assert shards == [[0, 4], [1, 5], [2], [3]]
    assert sorted(index for shard in shards for index in shard) == list(range(6))


def test_rank_metrics_use_global_branch_weighting_but_keep_reduced_grad_norms():
    result = weighted_mean_dict(
        [
            {"policy_loss": 2.0, "policy_grad_norm": 7.0},
            {"policy_loss": 8.0, "policy_grad_norm": 7.0},
        ],
        [2, 1],
        shared_keys={"policy_grad_norm"},
    )
    assert result == {"policy_loss": 4.0, "policy_grad_norm": 7.0}


def test_distributed_base_sampler_isolates_hidden_global_rng(tmp_path):
    class FakeRuntime:
        device = torch.device("cpu")

        def rollout_group(self, *args, **kwargs):
            return tmp_path, [float(torch.rand(()).item())]

    sampler = VideoTrajectorySampler(FakeRuntime())
    prepared = SimpleNamespace()
    torch.manual_seed(999)
    expected_after = torch.rand(())
    torch.manual_seed(999)
    first = sampler.sample_base(
        prepared, initial_seed=123, output_dir=tmp_path / "a", prompt="test"
    )
    after = torch.rand(())
    second = sampler.sample_base(
        prepared, initial_seed=123, output_dir=tmp_path / "b", prompt="test"
    )
    assert first == second
    assert torch.equal(after, expected_after)
