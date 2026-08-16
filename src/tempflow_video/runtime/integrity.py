from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .upstream_loader import PINNED_AWM_COMMIT, upstream_root


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def audit_upstream(value: str | Path | None = None) -> dict[str, object]:
    root = upstream_root(value)
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    status = _git(root, "status", "--porcelain=v1", "-z")
    report = {"root": str(root), "head": head, "status_sha256": sha256_bytes(status), "clean": not status}
    if head != PINNED_AWM_COMMIT:
        raise RuntimeError(f"upstream HEAD {head} != pinned {PINNED_AWM_COMMIT}")
    if status:
        raise RuntimeError("pinned AWM upstream is not clean")
    return report

