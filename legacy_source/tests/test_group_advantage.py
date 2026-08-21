import pytest
import torch

from experiments.tempflow_video.advantage import (
    fuse_psnr_sobel_rewards,
    standardize_group_rewards,
)


def test_group_advantage_is_population_standardized():
    result = standardize_group_rewards([1.0, 2.0, 3.0, 4.0], epsilon=1.0e-12)

    assert result.zero_std is False
    assert result.reward_mean == pytest.approx(2.5)
    assert result.reward_std == pytest.approx(1.11803398875)
    assert result.advantages.mean().item() == pytest.approx(0.0, abs=1.0e-12)
    assert result.advantages.std(unbiased=False).item() == pytest.approx(1.0, rel=1.0e-10)


def test_zero_variance_group_has_no_fabricated_signal():
    result = standardize_group_rewards([0.25, 0.25])

    assert result.zero_std is True
    assert torch.equal(result.advantages, torch.zeros(2, dtype=torch.float64))
    assert result.metrics()["zero_std_group_ratio"] == 1.0


def test_psnr_sobel_fusion_uses_separate_group_scales():
    fused, metrics = fuse_psnr_sobel_rewards(
        [20.0, 20.2, 20.1],
        [-0.030, -0.020, -0.025],
        psnr_weight=0.8,
        sobel_weight=0.2,
        psnr_scale=None,
        sobel_scale=None,
        psnr_scale_floor=0.005,
        sobel_scale_floor=0.0005,
    )

    assert metrics["psnr_scale_source"] == "group_std_with_floor"
    assert metrics["sobel_scale_source"] == "group_std_with_floor"
    assert metrics["psnr_scale"] == pytest.approx(metrics["psnr_group_std_db"])
    assert metrics["sobel_scale"] == pytest.approx(metrics["sobel_group_std"])
    assert int(torch.argmax(fused).item()) == 1


def test_psnr_sobel_fusion_honors_fixed_scales_and_normalizes_weights():
    fused, metrics = fuse_psnr_sobel_rewards(
        [20.0, 20.1],
        [-0.02, -0.01],
        psnr_weight=8.0,
        sobel_weight=2.0,
        psnr_scale=0.2,
        sobel_scale=0.01,
    )

    expected = 0.8 * torch.tensor([20.0, 20.1], dtype=torch.float64) / 0.2
    expected += 0.2 * torch.tensor([-0.02, -0.01], dtype=torch.float64) / 0.01
    assert torch.allclose(fused, expected)
    assert metrics["psnr_weight"] == pytest.approx(0.8)
    assert metrics["sobel_weight"] == pytest.approx(0.2)
    assert metrics["psnr_scale_source"] == "fixed"


@pytest.mark.parametrize("rewards", [[1.0], [1.0, float("nan")]])
def test_invalid_groups_are_rejected(rewards):
    with pytest.raises(ValueError):
        standardize_group_rewards(rewards)
