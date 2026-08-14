"""experiments.action_following 包。

默认不 import 任何重量级依赖（torch / cv2 / SAM3 / CoWTracker），
模块经 __getattr__ 懒加载，保证开发机（无 GPU）可纯 numpy 运行
metrics_fdce / fdce_tracks / aggregate / action_command / metrics_action_following，
GPU 模块（sam_tracking / cowtracker_tracking / yolo_detector）在目标环境按需加载。
"""

from __future__ import annotations

import importlib

_LAZY_MODULES = (
    "action_command",
    "aggregate",
    "cowtracker_tracking",
    "fdce_tracks",
    "metrics_action_following",
    "metrics_fdce",
    "viz_fdce",
    "sam_tracking",
    "yolo_detector",
)


def __getattr__(name):
    if name in _LAZY_MODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_MODULES))
