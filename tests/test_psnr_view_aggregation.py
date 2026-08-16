from tempflow_video.rewards.psnr_reward import aggregate_views


def test_psnr_view_aggregation():
    value, mean, worst = aggregate_views({"head": 20, "hand_left": 22, "hand_right": 24})
    assert mean == 22 and worst == 20 and value == 21.2

