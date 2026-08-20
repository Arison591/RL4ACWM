import pytest

from experiments.tempflow_video.run import _minimum_group_std, _training_reward


def _config(mode="fdce_only_raw_error"):
    return {
        "reward_fusion": {"mode": mode, "min_group_std": 0.002},
        "tempflow": {"zero_std_threshold": 1.0e-8},
    }


def test_fdce_training_uses_unclipped_negative_raw_error():
    result = {
        "total_reward": 0.0,
        "action_metrics": {"fdce": 17.5},
    }

    assert _training_reward(result, _config()) == -17.5


def test_fdce_training_rejects_missing_metric():
    with pytest.raises(ValueError, match="no raw FDCE"):
        _training_reward({"total_reward": None, "action_metrics": {}}, _config())


def test_generic_minimum_group_std_takes_precedence():
    assert _minimum_group_std(_config()) == 0.002
