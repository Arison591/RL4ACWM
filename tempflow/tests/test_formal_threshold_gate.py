import pytest
from tempflow_video.rewards.component_advantage import component_advantages

def test_formal_training_refuses_null_thresholds():
    with pytest.raises(RuntimeError):
        component_advantages([0, 1], [20, 21], action_min_group_std=None,
                             psnr_min_group_std_db=None, formal_training=True)

