import importlib.util
import os
import sys
from pathlib import Path

import torch
import pytest

from tempflow_video.core import transitions as new


def _old_module():
    source = Path(__file__).resolve().parents[1] / "legacy_source/experiments/tempflow_video/dynamics.py"
    if not source.exists():
        value = os.environ.get("ORIGINAL_REPO_ROOT")
        if not value:
            pytest.skip("legacy transition source unavailable")
        source = Path(value) / "experiments/tempflow_video/dynamics.py"
    spec = importlib.util.spec_from_file_location("legacy_dynamics", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_legacy_transition_parity():
    old = _old_module()
    sample = torch.randn(2, 3, 2, 4, dtype=torch.float64)
    velocity = torch.randn_like(sample)
    noise = torch.randn_like(sample)
    kwargs = dict(flow_time=0.7, next_flow_time=0.6, eta=0.7, exploration_noise=noise)
    left, right = old.edm_sde_transition_with_logprob(sample, velocity, **kwargs), new.edm_sde_transition_with_logprob(sample, velocity, **kwargs)
    for key in ("mean", "std", "next_sample", "log_prob", "rf_noise_std"):
        torch.testing.assert_close(getattr(left, key), getattr(right, key), rtol=0, atol=0)
