from tempflow_video.adapters.legacy_reward_adapter import LegacyRewardAdapter


def test_legacy_reward_adapter_forwards_without_transform():
    seen = {}
    expected = {"total_reward": 0.4, "action_reward": 0.2, "geometry": {"metrics": {"balanced_psnr_db": 21.0}}}
    def fake(gt, pred, **kwargs):
        seen.update(gt=gt, pred=pred, kwargs=kwargs)
        return expected
    result = LegacyRewardAdapter({}, compute_fn=fake).score("gt.mp4", "pred.mp4", max_frames=29)
    assert result is expected
    assert seen == {"gt": "gt.mp4", "pred": "pred.mp4", "kwargs": {"max_frames": 29}}

