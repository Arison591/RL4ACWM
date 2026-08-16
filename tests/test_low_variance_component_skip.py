from tempflow_video.rewards.component_advantage import component_advantages


def test_low_variance_psnr_skip_preserves_action():
    out = component_advantages([0, 1], [20, 20.00001], action_min_group_std=0.1,
                               psnr_min_group_std_db=0.001)
    assert out.psnr.skipped and not out.action.skipped
    assert out.psnr.advantages.tolist() == [0, 0]

