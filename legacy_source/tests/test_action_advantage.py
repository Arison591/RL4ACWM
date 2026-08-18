import pytest

from experiments.tempflow_video.action_advantage import build_action_advantages


def _input(**overrides):
    value = {
        "command_raw_error": [1.0, 2.0, 4.0, 8.0],
        "fdce_reward": [0.9, 0.1, 0.8, 0.2],
        "iou_reward": [0.1, 0.9, 0.2, 0.8],
        "valid_arms": [2, 2, 2, 2],
        "coverage": [1.0, 1.0, 1.0, 1.0],
        "noise_floors": {"command": 0.01, "fdce": 0.01, "iou": 0.01},
        "weights": {"command": 0.7, "fdce": 0.2, "iou": 0.1},
    }
    value.update(overrides)
    return value


def test_command_is_required_even_when_auxiliary_components_vary():
    result = build_action_advantages(**_input(command_raw_error=[1.0, 1.0, 1.0, 1.0]))
    assert result.advantages is None
    assert result.skip_reason == "command_not_informative"
    assert result.metrics["fdce_gate"] == 1.0


def test_missing_arm_skips_group_before_auxiliary_reward_can_update():
    result = build_action_advantages(**_input(valid_arms=[2, 1, 2, 2]))
    assert result.advantages is None
    assert result.metrics["command_complete"] == 0.0


def test_components_are_standardized_independently_without_second_zscore():
    result = build_action_advantages(**_input())
    assert result.skip_reason is None
    assert result.advantages.mean().item() == pytest.approx(0.0, abs=1e-12)
    assert (result.advantages > 0).any()
    assert (result.advantages < 0).any()
    assert result.metrics["mean_abs_weighted_command"] > result.metrics["mean_abs_weighted_fdce"]
    assert result.metrics["mean_abs_weighted_command"] > result.metrics["mean_abs_weighted_iou"]


def test_auxiliary_component_below_noise_floor_is_zeroed_not_command():
    result = build_action_advantages(**_input(noise_floors={"command": 0.01, "fdce": 1.0, "iou": 1.0}))
    assert result.skip_reason is None
    assert result.metrics["command_gate"] == 1.0
    assert result.metrics["fdce_gate"] == 0.0
    assert result.metrics["iou_gate"] == 0.0
