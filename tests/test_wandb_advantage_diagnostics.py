from __future__ import annotations

import pytest

from experiments.awm_coca.wandb_monitor import _advantage_diagnostics


def test_advantage_diagnostics_detect_complete_cancellation():
    metrics = _advantage_diagnostics(
        [
            {"total_reward": 0.5, "action_reward": 1.0, "geometry_reward": 0.0},
            {"total_reward": 0.5, "action_reward": 0.0, "geometry_reward": 1.0},
        ],
        action_weight=0.5,
        geometry_weight=0.5,
    )

    assert metrics["advantage/action_std"] == pytest.approx(1.0)
    assert metrics["advantage/geometry_std"] == pytest.approx(1.0)
    assert metrics["advantage/action_geometry_corr"] == pytest.approx(-1.0)
    assert metrics["advantage/action_flip_rate"] == pytest.approx(0.0)
    assert metrics["advantage/cancellation_rate"] == pytest.approx(1.0)


def test_advantage_diagnostics_detect_action_sign_flips():
    metrics = _advantage_diagnostics(
        [
            {"total_reward": 0.3, "action_reward": 0.6, "geometry_reward": 0.0},
            {"total_reward": 0.7, "action_reward": 0.4, "geometry_reward": 1.0},
        ],
        action_weight=0.5,
        geometry_weight=0.5,
    )

    assert metrics["advantage/action_geometry_corr"] == pytest.approx(-1.0)
    assert metrics["advantage/action_flip_rate"] == pytest.approx(1.0)
    assert metrics["advantage/cancellation_rate"] == pytest.approx(1.0 / 3.0)


def test_advantage_diagnostics_ignore_unpaired_or_insufficient_rewards():
    assert _advantage_diagnostics(
        [
            {"total_reward": 0.55, "action_reward": 0.6, "geometry_reward": 0.5},
            {"total_reward": None, "action_reward": 0.4, "geometry_reward": None},
        ],
        action_weight=0.5,
        geometry_weight=0.5,
    ) == {}


def test_advantage_diagnostics_use_same_valid_seed_set_as_training():
    metrics = _advantage_diagnostics(
        [
            {"total_reward": 0.5, "valid": True, "action_reward": 1.0, "geometry_reward": 0.0},
            {"total_reward": 0.5, "valid": True, "action_reward": 0.0, "geometry_reward": 1.0},
            {"total_reward": None, "valid": False, "action_reward": 100.0, "geometry_reward": 100.0},
        ],
        action_weight=0.5,
        geometry_weight=0.5,
    )

    assert metrics["advantage/action_std"] == pytest.approx(1.0)
    assert metrics["advantage/geometry_std"] == pytest.approx(1.0)
    assert metrics["advantage/action_geometry_corr"] == pytest.approx(-1.0)
