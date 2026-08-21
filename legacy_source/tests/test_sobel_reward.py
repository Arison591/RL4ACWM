import cv2
import numpy as np
import pytest

from experiments.tempflow_video.reward_adapter import _frame_sobel_error


def _edge_frame() -> np.ndarray:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, 16:] = 255
    return frame


def test_sobel_error_is_zero_for_matching_frames():
    frame = _edge_frame()
    error = _frame_sobel_error(frame, frame.copy(), charbonnier_epsilon=1.0e-3)
    assert error == pytest.approx(0.0, abs=1.0e-8)


def test_sobel_error_penalizes_blur_against_gt_edges():
    gt = _edge_frame()
    blurred = cv2.GaussianBlur(gt, (9, 9), 2.0)
    assert _frame_sobel_error(
        gt, blurred, charbonnier_epsilon=1.0e-3
    ) > _frame_sobel_error(gt, gt, charbonnier_epsilon=1.0e-3)


def test_sobel_error_penalizes_unmatched_high_frequency_texture():
    gt = _edge_frame()
    checker = ((np.indices((32, 32)).sum(axis=0) % 2) * 255).astype(np.uint8)
    pred = np.repeat(checker[:, :, None], 3, axis=2)
    assert _frame_sobel_error(
        gt, pred, charbonnier_epsilon=1.0e-3
    ) > _frame_sobel_error(gt, gt, charbonnier_epsilon=1.0e-3)
