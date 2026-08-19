from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

PINNED_AWM_COMMIT = "dce69e48a952449e873a791812e506df878bc8a9"


def upstream_root(value: str | Path | None = None) -> Path:
    raw = value or os.environ.get("AWM_UPSTREAM_ROOT")
    if not raw:
        raise RuntimeError("AWM_UPSTREAM_ROOT must point to the pinned clean checkout")
    root = Path(raw).expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"AWM_UPSTREAM_ROOT is not a Git checkout: {root}")
    return root


def install_upstream_import_path(value: str | Path | None = None) -> Path:
    root = upstream_root(value)
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


def import_upstream(module: str, value: str | Path | None = None):
    install_upstream_import_path(value)
    return importlib.import_module(module)

