from experiments.tempflow_video.run import _build_correction_schedule


def test_correction_schedule_covers_16_by_3_grid_in_balanced_four_group_windows():
    timesteps = [2, 6, 10]
    first = _build_correction_schedule(condition_count=16, timesteps=timesteps, seed=42)
    second = _build_correction_schedule(condition_count=16, timesteps=timesteps, seed=42)

    assert first == second
    assert len(first) == 48
    assert set(first) == {(condition, timestep) for condition in range(16) for timestep in timesteps}
    for start in range(0, len(first), 4):
        window = first[start:start + 4]
        assert len({condition for condition, _ in window}) == 4
        assert set(timesteps).issubset({timestep for _, timestep in window})
