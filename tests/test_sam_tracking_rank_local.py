import os

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
