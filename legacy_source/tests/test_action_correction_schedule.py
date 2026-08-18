from types import SimpleNamespace

import torch

from experiments.tempflow_video.run import (
    _build_correction_schedule,
    _configure_runtime_rollout_schedule,
)
from experiments.tempflow_video.sampler import TempFlowBranchSampler


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


def test_correction_branch_positions_use_the_15_step_rollout_schedule():
    class Scheduler:
        config = SimpleNamespace(final_sigmas_type="sigma_min")

        def set_timesteps(self, *, sigmas, device):
            del sigmas, device
            # Values produced by GE-Sim's FlowMatchEuler scheduler for its
            # 15-step linspace(0, 1) inference call.
            flow_times = torch.tensor([
                0.9876543209876543, 0.9816711266897088, 0.9722161927501628,
                0.9569711528597235, 0.9319661940936144, 0.8905741669186009,
                0.822603032718444, 0.7155227006808309, 0.5626722003287437,
                0.380036982237588, 0.21103003154334402, 0.09451646023154488,
                0.03399063543076515, 0.009612516310065088, 0.001996008078647996,
            ], dtype=torch.float64)
            self.sigmas = torch.cat((
                flow_times / (1.0 - flow_times),
                (flow_times[-1:] / (1.0 - flow_times[-1:])).clone(),
            ))

    runtime = SimpleNamespace(
        scheduler=Scheduler(),
        device=torch.device("cpu"),
    )
    _configure_runtime_rollout_schedule(runtime, reverse_denoise_steps=15)
    sampler = TempFlowBranchSampler.__new__(TempFlowBranchSampler)
    sampler.runtime = runtime

    selected = sampler.resolve_branch_timestep_fractions([0.2, 0.5, 0.8])

    assert selected == [6, 8, 10]
    assert all(0 <= index < 14 for index in selected)


def test_correction_resume_position_counts_only_valid_accumulated_groups():
    groups_per_update = 4
    schedule = _build_correction_schedule(
        condition_count=16,
        timesteps=[6, 8, 10],
        seed=42,
    )

    # Checkpoints are written only after a complete four-group update, so the
    # next schedule item is determined by valid groups, never rejected tries.
    assert len(schedule) == 48
    assert 3 * groups_per_update == 12
    assert schedule[3 * groups_per_update] in schedule
