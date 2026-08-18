import numpy as np

from experiments.awm_coca import reward_runner


def test_action_metrics_use_matching_arm_command(monkeypatch):
    pred = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    commands = {
        "left": np.zeros((3, 2), dtype=np.float32),
        "right": np.full((3, 2), 10.0, dtype=np.float32),
    }

    def fake_track(video, arm):
        value = 0.0 if arm == "left" else 10.0
        return np.full((len(video), 2), value, dtype=np.float32), np.ones(len(video), dtype=bool)

    def fake_metric(traj, command, *, diag):
        return {"af_ate": float(np.abs(traj - command).mean()), "af_ate_norm": 0.0}

    monkeypatch.setattr(reward_runner.yolo_detector, "track_pred", fake_track)
    monkeypatch.setattr(reward_runner, "action_following_metrics", fake_metric)

    metrics = reward_runner._arm_specific_action_metrics(pred, commands, diag=1.0)

    assert len(metrics) == 2
    assert [row["af_ate"] for row in metrics] == [0.0, 0.0]
