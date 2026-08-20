import numpy as np

from experiments.action_following.sam_tracking import SAMMaskDiagnostics
from experiments.awm_coca import reward_runner


def test_fdce_only_reward_skips_unrelated_action_detectors(monkeypatch):
    video = np.zeros((3, 8, 10, 3), dtype=np.uint8)
    masks = np.ones((3, 8, 10), dtype=bool)
    diagnostic = SAMMaskDiagnostics(True, False, None, 3, 3)

    monkeypatch.setattr(
        reward_runner,
        "prepare_video_pair",
        lambda *args, **kwargs: (video, video, {"num_frames": 3}),
    )
    monkeypatch.setattr(
        reward_runner,
        "_track_sam_pair",
        lambda *args, **kwargs: (
            (masks, masks),
            {"reference": diagnostic.__dict__, "generated": diagnostic.__dict__},
        ),
    )
    monkeypatch.setattr(
        reward_runner,
        "_fdce_only_metrics",
        lambda *args, **kwargs: {"fdce": 2.5, "fdce_valid": True},
    )
    monkeypatch.setattr(
        reward_runner,
        "compute_all",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("IoU path called")),
    )
    monkeypatch.setattr(
        reward_runner,
        "_fdce_metrics",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("command path called")),
    )

    result = reward_runner.compute_head_reward(
        "gt.mp4",
        "pred.mp4",
        reward_mode="action",
        # Config inheritance preserves the base keys at zero weight.  They
        # must not force the mixed IoU/command detector path back on.
        action_metric_weights={
            "mean_iou": 0.0,
            "af_fdce_ate_norm": 0.0,
            "fdce": 1.0,
        },
        fdce_scale=10.0,
        prep_dir=None,
    )

    assert result["valid"] is True
    assert result["action_metrics"]["fdce"] == 2.5
    assert result["action_reward"] == 0.75
    assert result["total_reward"] == 0.75
