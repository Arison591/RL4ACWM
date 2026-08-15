import os

import numpy as np

from experiments.action_following import sam_tracking
from experiments.action_following.sam_tracking import (
    _force_rank_local_inference,
    _rank_local_sam3_environment,
)


class _Detector:
    rank = 2
    world_size = 4


class _Model:
    rank = 2
    world_size = 4
    detector = _Detector()


def test_force_rank_local_inference_ignores_torchrun_world() -> None:
    model = _force_rank_local_inference(_Model())

    assert model.rank == 0
    assert model.world_size == 1
    assert model.detector.rank == 0
    assert model.detector.world_size == 1


def test_rank_local_environment_masks_and_restores_torchrun_values(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")

    with _rank_local_sam3_environment():
        assert os.environ["RANK"] == "0"
        assert os.environ["WORLD_SIZE"] == "1"

    assert os.environ["RANK"] == "3"
    assert os.environ["WORLD_SIZE"] == "4"


def test_track_masks_wraps_full_sam3_pass_in_bf16_autocast(monkeypatch) -> None:
    state = {"autocast": False, "entered": 0}

    class _Autocast:
        def __enter__(self):
            state["autocast"] = True
            state["entered"] += 1

        def __exit__(self, *_args):
            state["autocast"] = False

    class _VideoModel:
        score_threshold_detection = 0.5
        new_det_thresh = 0.7

        def init_state(self, _frames):
            assert state["autocast"] is True
            return {"orig_height": 2, "orig_width": 3, "num_frames": 1}

        def add_prompt(self, _inference_state, *, frame_idx, text_str):
            assert frame_idx == 0
            assert text_str == "robot arm"
            assert state["autocast"] is True

        def propagate_in_video(self, _inference_state):
            assert state["autocast"] is True
            yield 0, {"out_binary_masks": np.ones((1, 2, 3), dtype=bool)}

    monkeypatch.setattr(sam_tracking, "get_sam3_video_model", lambda: _VideoModel())
    monkeypatch.setattr(sam_tracking, "_sam3_inference_autocast", lambda: _Autocast())

    masks = sam_tracking.track_masks(np.zeros((1, 2, 3, 3), dtype=np.uint8))

    assert state == {"autocast": False, "entered": 1}
    assert masks.shape == (1, 2, 3)
    assert masks.all()
