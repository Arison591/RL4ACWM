import numpy as np

from tempflow_video.rewards.psnr_reward import compute_psnr_reward


def test_psnr_frame_scope_excludes_runtime_history():
    gt = np.zeros((6, 2, 2, 3), dtype=np.uint8)
    pred_a, pred_b = gt.copy(), gt.copy()
    pred_a[:2] = 1
    pred_b[:2] = 100
    a = compute_psnr_reward({"head": gt}, {"head": pred_a}, history_frames=2)
    b = compute_psnr_reward({"head": gt}, {"head": pred_b}, history_frames=2)
    assert a.psnr_aggregate_future_db == b.psnr_aggregate_future_db
    assert a.psnr_aggregate_full_db != b.psnr_aggregate_full_db

