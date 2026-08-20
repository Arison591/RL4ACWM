from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from da3_reward import DA3RewardConfig, compute_da3_gt_reward
from experiments.tempflow_video import da3_video_reward
from experiments.tempflow_video.da3_video_reward import DA3MonoVideoReward
from experiments.tempflow_video.reward_adapter import VideoRewardAdapter
from experiments.tempflow_video.run import _group_std_threshold, _training_reward


class _FakeDA3Runner:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def infer_mono(self, images):
        self.calls.append(len(images))
        depth = torch.ones(len(images), 8, 8)
        confidence = torch.ones_like(depth)
        return depth, confidence

    def metadata(self):
        return {"backend": "fake-da3", "offline": True}


def test_da3_core_reward_is_negative_mean_view_depth_error():
    target = torch.linspace(1.0, 2.0, 2 * 2 * 16 * 16).reshape(2, 2, 16, 16)
    generated = target.clone()
    generated[1, :, 4:12, 4:12] += 0.4
    result = compute_da3_gt_reward(
        target,
        generated,
        config=DA3RewardConfig(
            trim_fraction=0.0,
            dynamic_weight=0.0,
            confidence_mode="none",
            border_fraction=0.0,
            max_alignment_samples=1024,
            max_reduction_samples=1024,
        ),
        include_per_frame=False,
    )

    assert result.error == pytest.approx(result.per_view_error.mean().item())
    assert result.reward == pytest.approx(-result.error)
    assert result.per_view_error[0].item() == pytest.approx(0.0, abs=1.0e-7)
    assert result.per_view_error[1].item() > 0.0


def test_tempflow_da3_uses_same_three_views_and_future_range_as_psnr(monkeypatch, tmp_path):
    calls: list[tuple[Path, int, int]] = []

    def fake_future_frames(path: Path, start: int, end: int):
        calls.append((Path(path), start, end))
        frames = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(end - start + 1)]
        return frames, np.zeros((12, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(da3_video_reward, "_future_frames", fake_future_frames)
    runner = _FakeDA3Runner()
    reward = DA3MonoVideoReward(
        {
            "geometry_cameras": ["head", "hand_left", "hand_right"],
            "geometry_future_start": 4,
            "geometry_future_end": 28,
            "gt_video_templates": {
                camera: str(tmp_path / "gt" / "{condition_id}" / f"{camera}.mp4")
                for camera in ("head", "hand_left", "hand_right")
            },
        },
        runner=runner,
    )

    result = reward.score_paths(
        condition_id="condition-1", prediction_dir=tmp_path / "prediction"
    )

    assert result["valid"] is True
    assert result["total_reward"] == pytest.approx(0.0)
    assert result["geometry"]["metrics"]["frame_count_per_view"] == 25
    assert result["geometry"]["metrics"]["inference_mode"] == "mono_per_frame_independent"
    assert all((start, end) == (4, 28) for _, start, end in calls)
    assert {path.name for path, _, _ in calls} == {
        "head.mp4",
        "hand_left.mp4",
        "hand_right.mp4",
        "head_color.mp4",
        "hand_left_color.mp4",
        "hand_right_color.mp4",
    }
    assert runner.calls == [25, 25, 25, 25, 25, 25]

    # GT pseudo-depth is cached per process/condition; generated clips are not.
    reward.score_paths(condition_id="condition-1", prediction_dir=tmp_path / "prediction-2")
    assert len(calls) == 9
    assert runner.calls == [25] * 9


def test_da3_training_reward_and_variance_gate_are_not_psnr_specific():
    config = {
        "reward_fusion": {"mode": "da3_mono_only", "min_group_std": 1.0e-5},
        "tempflow": {"zero_std_threshold": 1.0e-8},
    }
    result = {"total_reward": -0.3, "da3_mono": {"reward": -0.2}}

    assert _training_reward(config, result) == pytest.approx(-0.2)
    assert _group_std_threshold(config) == pytest.approx(1.0e-5)


def test_video_reward_adapter_dispatches_da3_without_legacy_scorer(tmp_path):
    adapter = VideoRewardAdapter(
        {
            "evaluator": "da3_mono",
            "time_alignment_protocol": "legacy_frame_index_truncate",
            "geometry_cameras": ["head"],
            "geometry_future_start": 4,
            "geometry_future_end": 28,
            "gt_video_templates": {"head": str(tmp_path / "{condition_id}.mp4")},
            "da3_source_root": str(tmp_path / "source"),
        }
    )

    class _FakeEvaluator:
        def score_paths(self, **kwargs):
            return {"valid": True, "total_reward": -0.25, "kwargs": kwargs}

    adapter._da3 = _FakeEvaluator()
    result = adapter.score_paths(
        condition_id="condition-1",
        prep_dir=str(tmp_path / "unused-prep"),
        prediction_dir=tmp_path / "prediction",
    )

    assert result["total_reward"] == pytest.approx(-0.25)
    assert result["kwargs"] == {
        "condition_id": "condition-1",
        "prediction_dir": tmp_path / "prediction",
    }
