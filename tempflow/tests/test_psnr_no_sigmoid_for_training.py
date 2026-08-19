from tempflow_video.rewards.component_advantage import component_advantages


def test_psnr_no_sigmoid_for_training():
    out = component_advantages([0.1, 0.2], [18.0, 22.0], action_min_group_std=0,
                               psnr_min_group_std_db=0, formal_training=True)
    assert out.psnr.rewards.tolist() == [18.0, 22.0]

