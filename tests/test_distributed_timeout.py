from datetime import timedelta

import pytest

from experiments.awm_coca.condition_sampler import ResumableConditionSampler
from experiments.awm_coca.run_train import (
    _assert_group_alignment,
    _distributed_timeout,
    _group_skip_reason,
    _sampler_state_after_groups,
)


def test_distributed_timeout_defaults_to_two_hours(monkeypatch):
    monkeypatch.delenv("DIST_TIMEOUT_SECONDS", raising=False)
    assert _distributed_timeout() == timedelta(hours=2)


def test_distributed_timeout_can_be_overridden(monkeypatch):
    monkeypatch.setenv("DIST_TIMEOUT_SECONDS", "3600")
    assert _distributed_timeout() == timedelta(hours=1)


def test_distributed_timeout_must_be_positive(monkeypatch):
    monkeypatch.setenv("DIST_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        _distributed_timeout()


def test_group_skips_synchronously_when_one_rank_has_no_valid_seed():
    gathered = [
        [(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4)],
        [(4, 0.1), (5, 0.2), (6, 0.3), (7, 0.4)],
        [(8, 0.1), (9, 0.2), (10, 0.3), (11, 0.4)],
        [],
    ]

    reason = _group_skip_reason(gathered, min_valid_seeds=8)

    assert reason == "no valid seed on rank(s) [3]"


def test_group_trains_when_threshold_and_every_rank_are_satisfied():
    gathered = [
        [(0, 0.1), (1, 0.2)],
        [(2, 0.1), (3, 0.2)],
        [(4, 0.1), (5, 0.2)],
        [(6, 0.1), (7, 0.2)],
    ]

    assert _group_skip_reason(gathered, min_valid_seeds=8) is None


def test_group_skips_when_global_valid_count_is_below_threshold():
    gathered = [[(0, 0.1)], [(1, 0.2)], [(2, 0.3)], [(3, 0.4)]]

    assert _group_skip_reason(gathered, min_valid_seeds=8) == (
        "only 4 valid seeds (need >= 8)"
    )


def test_checkpoint_sampler_state_ignores_prefetched_position():
    sampler = ResumableConditionSampler(203, seed=42)
    sampler.epoch = 9
    sampler.position = 120  # DataLoader may have prefetched far ahead.

    state = _sampler_state_after_groups(sampler, 417)

    assert state["epoch"] == 2
    assert state["position"] == 11


def test_group_alignment_rejects_different_conditions(monkeypatch):
    monkeypatch.setattr(
        "experiments.awm_coca.run_train._gather_per_rank",
        lambda value: [value, ("different", value[1], value[2])],
    )

    with pytest.raises(RuntimeError, match="alignment mismatch"):
        _assert_group_alignment("condition", 3, 9)
