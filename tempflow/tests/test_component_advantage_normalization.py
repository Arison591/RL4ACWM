from tempflow_video.rewards.component_advantage import component_advantages


def test_component_advantage_normalization_handles_scale_gap():
    out = component_advantages([0, 10, 20], [20, 20.1, 20.2], action_min_group_std=0,
                               psnr_min_group_std_db=0, advantage_clip=10)
    assert abs(float(out.action.advantages.std(unbiased=False)) - 1) < 2e-5
    assert abs(float(out.psnr.advantages.std(unbiased=False)) - 1) < 2e-5

