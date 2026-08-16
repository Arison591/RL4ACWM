import os

import pytest

from tempflow_video.runtime.upstream_loader import import_upstream


def test_clean_upstream_import():
    if not os.environ.get("AWM_UPSTREAM_ROOT"):
        pytest.skip("AWM_UPSTREAM_ROOT not configured")
    module = import_upstream("experiments.awm_coca.reward_functions")
    assert callable(module.compute_geometry_reward)

