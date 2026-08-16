import copy

from experiments.tempflow_video import reward_adapter as adapter_module
from experiments.tempflow_video.reward_adapter import VideoRewardAdapter


def _reward_config(tmp_path):
    template = str(tmp_path / "gt" / "{condition_id}" / "{camera}_29_frames.mp4")
    return {
        "gt_video_template": str(tmp_path / "gt" / "{condition_id}" / "head_29_frames.mp4"),
        "gt_video_templates": {
            camera: template for camera in ("head", "hand_left", "hand_right")
        },
        "mode": "joint",
        "max_frames": 29,
        "segmentation_prompt": "robot arm",
        "confidence": 0.1,
        "action_metric_weights": {
            "mean_iou": 0.1,
            "af_fdce_ate_norm": 0.7,
            "fdce": 0.2,
        },
        "af_fdce_ate_norm_scale": 0.2,
        "fdce_scale": 10.0,
        "fdce_k": 16,
        "fdce_visibility_threshold": 0.5,
        "fdce_min_visible_fraction": 0.8,
        "fdce_min_common_frames": 1,
        "fdce_seed": 0,
        "geometry_enabled": True,
        "geometry_cameras": ["head", "hand_left", "hand_right"],
        "geometry_future_start": 4,
        "geometry_future_end": 28,
        "geometry_mean_weight": 0.6,
        "geometry_worst_weight": 0.4,
        "geometry_psnr_center_db": 20.4,
        "geometry_psnr_temperature_db": 1.8,
        "action_weight": 0.5,
        "geometry_weight": 0.5,
    }


def test_adapter_forwards_legacy_reward_without_changing_protocol(monkeypatch, tmp_path):
    captured = {}
    expected = {
        "total_reward": 0.42,
        "action_reward": 0.31,
        "geometry_reward": 0.53,
        "geometry": {"metrics": {"per_view_psnr_db": {"head": 20.0}}},
        "valid": True,
    }

    def fake_legacy(gt_path, pred_path, **kwargs):
        captured.update({"gt_path": gt_path, "pred_path": pred_path, "kwargs": kwargs})
        return copy.deepcopy(expected)

    monkeypatch.setattr(adapter_module, "compute_head_reward", fake_legacy)
    config = _reward_config(tmp_path)
    adapter = VideoRewardAdapter(config)
    result = adapter.score_paths(
        condition_id="sample-a",
        prep_dir=str(tmp_path / "prep" / "sample-a"),
        prediction_dir=tmp_path / "pred",
    )

    assert result == expected
    assert captured["gt_path"].endswith("gt/sample-a/head_29_frames.mp4")
    assert captured["pred_path"].endswith("pred/head_color.mp4")
    assert captured["kwargs"]["max_frames"] == 29
    assert captured["kwargs"]["action_metric_weights"] == config["action_metric_weights"]
    assert captured["kwargs"]["geometry_cameras"] == config["geometry_cameras"]
    assert captured["kwargs"]["all_camera_videos"]["hand_right"]["pred"].endswith(
        "pred/hand_right_color.mp4"
    )


def test_reward_config_hash_is_order_independent(tmp_path):
    config = _reward_config(tmp_path)
    reversed_config = dict(reversed(list(config.items())))
    assert VideoRewardAdapter(config).config_sha256 == VideoRewardAdapter(reversed_config).config_sha256
