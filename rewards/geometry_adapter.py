"""Geometry-model boundary for the VGGRPO-style RGB reward.

The reward code consumes a small, model-independent geometry contract.  A real
4D geometry backend is responsible for preprocessing RGB frames and converting
its native output to the coordinate conventions documented by
``GeometryOutput``.  Tests can use ``MockGeometryAdapter`` without loading a
geometry model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor


@dataclass
class GeometryOutput:
    """Model-independent geometry for one video/view.

    Shapes and conventions:
        camera_poses: ``[T, 4, 4]`` camera-to-world transforms.
        depths: ``[T, H, W]`` positive camera-space z-depth.
        point_maps: ``[T, H, W, 3]`` points in world coordinates.  When scene
            flow is present, all frames use the same persistent reference grid.
        scene_flows: optional ``[T, H, W, 3]`` world-space displacement on the
            same persistent reference grid as ``point_maps``.
        valid_mask: ``[T, H, W]`` boolean mask on each target frame's depth
            pixel grid.  Reprojection/depth comparison uses this mask.
        intrinsics: optional ``[T, 3, 3]`` pinhole intrinsics in pixel units.
            Reprojection reward requires this field.  It is optional at the
            adapter boundary so a missing value can be reported as an invalid
            geometry result instead of crashing inference.
        point_valid_mask: optional ``[T, H, W]`` boolean mask on the persistent
            reference-frame point grid used by ``point_maps`` and
            ``scene_flows``.  If omitted, reward code falls back to
            ``valid_mask`` for backward-compatible mock adapters.
        diagnostics: backend-specific, lightweight diagnostic values.

    ``T`` must match the number of frames passed to ``GeometryAdapter.infer``.
    The adapter must convert any backend-specific coordinate convention to the
    one above; reward formulas must not guess tensor layouts or conventions.
    """

    camera_poses: Tensor
    depths: Tensor
    point_maps: Tensor
    scene_flows: Optional[Tensor]
    valid_mask: Tensor
    intrinsics: Optional[Tensor] = None
    point_valid_mask: Optional[Tensor] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class GeometryAdapter(ABC):
    """Frozen geometry inference interface used by the RGB reward.

    ``video`` has explicit shape ``[T, C, H, W]``.  Group, batch, and view
    dimensions are handled by the reward module, so each adapter call receives
    exactly one candidate and one view.
    """

    @abstractmethod
    def infer(self, video: Tensor) -> GeometryOutput:
        """Infer geometry for one RGB video without computing gradients."""


class MockGeometryAdapter(GeometryAdapter):
    """Return queued geometry outputs for deterministic reward-only tests.

    Outputs are consumed in order, matching the reward module's candidate/view
    traversal.  This keeps mock behavior explicit and makes per-view aggregation
    tests easy to reason about.
    """

    def __init__(self, outputs: Sequence[GeometryOutput]) -> None:
        if not outputs:
            raise ValueError("MockGeometryAdapter requires at least one output")
        self._outputs: List[GeometryOutput] = list(outputs)
        self._next_index = 0

    @property
    def num_remaining(self) -> int:
        """Number of queued outputs that have not been consumed."""

        return len(self._outputs) - self._next_index

    @torch.no_grad()
    def infer(self, video: Tensor) -> GeometryOutput:
        if video.ndim != 4:
            raise ValueError(
                "MockGeometryAdapter expects video shaped [T, C, H, W], "
                f"got {tuple(video.shape)}"
            )
        if self._next_index >= len(self._outputs):
            raise RuntimeError("MockGeometryAdapter output queue is exhausted")

        output = self._outputs[self._next_index]
        self._next_index += 1
        return output
