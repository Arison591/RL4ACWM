import os

import pytest

from tempflow_video.runtime.integrity import audit_upstream


def test_upstream_integrity():
    if not os.environ.get("AWM_UPSTREAM_ROOT"):
        pytest.skip("AWM_UPSTREAM_ROOT not configured")
    report = audit_upstream()
    assert report["clean"] is True

