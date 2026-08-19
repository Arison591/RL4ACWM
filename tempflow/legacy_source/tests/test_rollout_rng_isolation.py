from types import SimpleNamespace

import torch

from experiments.tempflow_video.sampler import VideoTrajectorySampler


def test_rollout_group_forks_hidden_global_rng(tmp_path):
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
