from experiments.action_following.sam_tracking import _force_rank_local_inference


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
