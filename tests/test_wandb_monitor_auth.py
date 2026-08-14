from __future__ import annotations

import sys
import types

from experiments.awm_coca import wandb_monitor


class _FakeRun:
    id = "test-run"
    url = "https://wandb.ai/test/awm-coca/runs/test-run"

    def define_metric(self, *args, **kwargs):
        return None

    def finish(self):
        return None


def test_bundled_credentials_are_used_without_environment(monkeypatch, tmp_path):
    calls = {}
    fake_wandb = types.ModuleType("wandb")

    def login(*, key, relogin):
        calls["login"] = {"key": key, "relogin": relogin}
        return True

    def init(**kwargs):
        calls["init"] = kwargs
        return _FakeRun()

    fake_wandb.login = login
    fake_wandb.init = init
    fake_wandb.Settings = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_MODE"):
        monkeypatch.delenv(name, raising=False)

    monitor = wandb_monitor.WandbMonitor({}, tmp_path)

    assert calls["login"]["key"] == wandb_monitor._BUNDLED_WANDB_API_KEY
    assert calls["login"]["relogin"] is True
    assert calls["init"]["entity"] == wandb_monitor._BUNDLED_WANDB_ENTITY
    assert calls["init"]["project"] == "awm-coca"
    assert calls["init"]["mode"] == "online"
    assert monitor.run is not None
    monitor.finish()


def test_legacy_short_entity_is_repaired(monkeypatch, tmp_path):
    calls = {}
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.login = lambda **kwargs: True
    fake_wandb.Settings = lambda **kwargs: kwargs

    def init(**kwargs):
        calls.update(kwargs)
        return _FakeRun()

    fake_wandb.init = init
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_ENTITY", "hrqian06")
    monkeypatch.delenv("WANDB_MODE", raising=False)

    monitor = wandb_monitor.WandbMonitor({}, tmp_path)

    assert calls["entity"] == wandb_monitor._BUNDLED_WANDB_ENTITY
    monitor.finish()
