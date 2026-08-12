from datetime import timedelta

import pytest

from experiments.awm_coca.run_train import _distributed_timeout


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
