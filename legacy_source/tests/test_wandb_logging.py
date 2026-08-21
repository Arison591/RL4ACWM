import json
import sys
from types import SimpleNamespace

from experiments.tempflow_video.run import (
    _init_wandb_run,
    _wandb_eval_payload,
    _wandb_log,
)


class _FakeRun:
    project = "test-project"
    id = "test-run"
    url = "https://wandb.example/test-run"

    def __init__(self):
        self.logged = []

    def log(self, payload, *, step):
        self.logged.append((payload, step))


def test_wandb_init_writes_auditable_status(monkeypatch, tmp_path):
    fake_run = _FakeRun()
    fake_wandb = SimpleNamespace(init=lambda **kwargs: fake_run)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_MODE", "offline")

    result = _init_wandb_run(
        {"experiment": {"name": "test"}}, tmp_path, enabled=True
    )

    assert result is fake_run
    status = json.loads((tmp_path / "wandb_run.json").read_text())
    assert status["run_id"] == "test-run"
    assert status["url"] == fake_run.url


def test_wandb_init_resumes_explicit_run(monkeypatch, tmp_path):
    fake_run = _FakeRun()
    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_RUN_ID", "test-run")
    monkeypatch.setenv("WANDB_RESUME", "must")

    result = _init_wandb_run({}, tmp_path, enabled=True)

    assert result is fake_run
    assert captured["id"] == "test-run"
    assert captured["resume"] == "must"


def test_wandb_group_logging_flattens_numeric_metrics():
    run = _FakeRun()

    _wandb_log(run, {"metrics": {"loss": 1.25}, "ignored": "text"}, step=7)

    payload, step = run.logged[0]
    assert step == 7
    assert payload["reward.metrics.loss"] == 1.25
    assert payload["trainer/optimizer_step"] == 7.0
    assert "reward.ignored" not in payload


def test_wandb_eval_payload_keeps_only_summary_metrics():
    payload = _wandb_eval_payload(
        {
            "reward_mean": 0.5,
            "reward_std": 0.1,
            "training_reward_mean": 42.0,
            "training_reward_std": 2.0,
            "component_statistics": {
                "reward.geometry.metrics.balanced_psnr_db": {"mean": 20.75},
                "reward.sobel.score": {"mean": -0.03},
                "reward.sobel.balanced_error": {"mean": 0.03},
                "reward.geometry.metrics.per_frame_psnr_db.head[0]": {"mean": 99.0},
            },
        }
    )

    assert payload["eval/balanced_psnr_db"] == 20.75
    assert payload["eval/sobel_score"] == -0.03
    assert payload["eval/sobel_error"] == 0.03
    assert not any("per_frame" in key for key in payload)
