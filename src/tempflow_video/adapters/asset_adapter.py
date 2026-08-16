from __future__ import annotations

import os
from pathlib import Path


def asset_root() -> Path:
    value = os.environ.get("AWM_ASSET_ROOT")
    if not value:
        raise RuntimeError("AWM_ASSET_ROOT is required; assets are never vendored")
    return Path(value).expanduser().resolve()


def apply_asset_environment() -> Path:
    root = asset_root()
    os.environ.setdefault("HF_HOME", str(root / "huggingface"))
    return root

