import numpy as np
import pytest

from experiments.action_following.action_command import action_following_metrics
from experiments.awm_coca.reward_functions import action_metrics_to_reward
from experiments.awm_coca.reward_runner import _aggregate_arm_command_metrics


def _command_tracks(offset=0.0, *, swapped=False, reversed_time=False):
    time = np.arange(6, dtype=np.float32)
    left = np.stack([time, time * 0], axis=1)
    right = np.stack([100 + time, time * 0], axis=1)
    translation = np.array([offset, 0.0], dtype=np.float32)
    tracks = {"left": left + translation, "right": right + translation}
    if swapped:
        tracks = {"left": right + translation, "right": left + translation}
    if reversed_time:
        tracks = {arm: value[::-1] for arm, value in tracks.items()}
    return {"left": left, "right": right}, tracks


def _metrics(commands, tracks):
    rows = []
    for arm in ("left", "right"):
        rows.append({"arm": arm, **action_following_metrics(
            tracks[arm], commands[arm], diag=200.0, skip_t0=False
        )})
    return _aggregate_arm_command_metrics(rows)


def _component(raw_error):
    reward, components = action_metrics_to_reward(
        {"mean_iou": 1.0, "af_fdce_ate_norm": raw_error / 200.0, "fdce": 0.0},
        metric_weights={"mean_iou": 0.0, "af_fdce_ate_norm": 1.0, "fdce": 0.0},
        af_fdce_ate_norm_scale=0.2,
    )
    return reward, components["components"]["af_fdce_ate_norm"]


def test_perfect_small_and_large_translation_are_ordered_smoothly():
    commands, perfect_tracks = _command_tracks()
    _, small_tracks = _command_tracks(2.0)
    _, large_tracks = _command_tracks(20.0)
    perfect = _metrics(commands, perfect_tracks)["combined_raw_command_error"]
    small = _metrics(commands, small_tracks)["combined_raw_command_error"]
    large = _metrics(commands, large_tracks)["combined_raw_command_error"]
    assert perfect == pytest.approx(0.0)
    assert small == pytest.approx(2.0)
    assert large == pytest.approx(20.0)
    assert _component(perfect)[1] > _component(small)[1] > _component(large)[1]


def test_swapped_and_time_reversed_arms_are_worse_than_correct_matching():
    commands, correct = _command_tracks()
    _, swapped = _command_tracks(swapped=True)
    _, reversed_tracks = _command_tracks(reversed_time=True)
    correct_error = _metrics(commands, correct)["combined_raw_command_error"]
    assert _metrics(commands, swapped)["combined_raw_command_error"] > correct_error + 90.0
    assert _metrics(commands, reversed_tracks)["combined_raw_command_error"] >= correct_error + 3.0


def test_missing_arm_is_explicit_and_cannot_be_complete_command_signal():
    commands, tracks = _command_tracks()
    row = {"arm": "left", **action_following_metrics(tracks["left"], commands["left"], diag=200.0)}
    result = _aggregate_arm_command_metrics([row])
    assert result["command_valid_arms"] == 1
    assert result["command_is_complete"] is False
    assert result["right_arm_raw_error"] is None


def test_same_video_metrics_are_stateless_when_reused():
    commands, tracks_a = _command_tracks(3.0)
    _, tracks_b = _command_tracks(11.0)
    first = _metrics(commands, tracks_a)
    _metrics(commands, tracks_b)
    second = _metrics(commands, tracks_a)
    assert second == first
