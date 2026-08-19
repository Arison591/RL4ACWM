import pytest
import torch

from experiments.tempflow_video.advantage import standardize_group_rewards


def test_group_advantage_is_population_standardized():
    result = standardize_group_rewards([1.0, 2.0, 3.0, 4.0], epsilon=1.0e-12)

    assert result.zero_std is False
    assert result.reward_mean == pytest.approx(2.5)
    assert result.reward_std == pytest.approx(1.11803398875)
    assert result.advantages.mean().item() == pytest.approx(0.0, abs=1.0e-12)
    assert result.advantages.std(unbiased=False).item() == pytest.approx(1.0, rel=1.0e-10)


def test_zero_variance_group_has_no_fabricated_signal():
    result = standardize_group_rewards([0.25, 0.25])

    assert result.zero_std is True
    assert torch.equal(result.advantages, torch.zeros(2, dtype=torch.float64))
    assert result.metrics()["zero_std_group_ratio"] == 1.0


@pytest.mark.parametrize("rewards", [[1.0], [1.0, float("nan")]])
def test_invalid_groups_are_rejected(rewards):
    with pytest.raises(ValueError):
        standardize_group_rewards(rewards)
