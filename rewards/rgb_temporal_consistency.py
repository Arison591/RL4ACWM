"""Independent RGB correspondence error for the geometry reward.

Any4D's camera, depth, point-map, and scene-flow heads are correlated.  This
small CPU-side check anchors their self-consistency score to observations from
the RGB video without adding another learned model to the training stack.
"""

from typing import Any, Dict, Tuple

import torch
from torch import Tensor


@torch.no_grad()
def compute_rgb_temporal_error(
    video: Tensor,
    *,
    resize_long_side: int = 160,
    topk_worst_pairs: int = 3,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Return a dimensionless forward/backward correspondence error.

    ``video`` is ``[T, C, H, W]`` in either ``[0, 1]`` or ``[-1, 1]``.  Dense
    Farneback flow is deliberately used here: it is deterministic, lightweight,
    and independent of Any4D.  Each adjacent selected-frame pair combines
    photometric warping, forward/backward cycle, and local flow-strain errors.
    """

    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError(
            "RGB temporal consistency expects [T, 3, H, W], got "
            f"{tuple(video.shape)}"
        )
    if video.shape[0] < 2:
        raise ValueError("RGB temporal consistency needs at least two frames")
    if resize_long_side <= 0:
        raise ValueError("resize_long_side must be positive")
    if topk_worst_pairs <= 0:
        raise ValueError("topk_worst_pairs must be positive")

    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "RGB temporal consistency requires OpenCV and NumPy"
        ) from error

    frames = video.detach().float().cpu()
    if float(frames.min()) < 0.0:
        frames = frames.add(1.0).mul(0.5)
    frames = frames.clamp(0.0, 1.0)
    gray = (
        0.2989 * frames[:, 0]
        + 0.5870 * frames[:, 1]
        + 0.1140 * frames[:, 2]
    )
    gray_u8 = gray.mul(255.0).round().to(torch.uint8).numpy()

    height, width = gray_u8.shape[-2:]
    scale = min(1.0, resize_long_side / max(height, width))
    target_width = max(8, int(round(width * scale)))
    target_height = max(8, int(round(height * scale)))
    if (target_height, target_width) != (height, width):
        gray_u8 = np.stack(
            [
                cv2.resize(
                    frame,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )
                for frame in gray_u8
            ]
        )

    grid_y, grid_x = np.mgrid[:target_height, :target_width].astype(np.float32)
    image_diagonal = float(np.hypot(target_height, target_width))
    pair_errors = []
    pair_diagnostics = []
    for first, second in zip(gray_u8[:-1], gray_u8[1:]):
        forward = cv2.calcOpticalFlowFarneback(
            first, second, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        backward = cv2.calcOpticalFlowFarneback(
            second, first, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        map_x = grid_x + forward[..., 0]
        map_y = grid_y + forward[..., 1]
        valid = (
            (map_x >= 1.0)
            & (map_x < target_width - 2.0)
            & (map_y >= 1.0)
            & (map_y < target_height - 2.0)
        )
        if int(valid.sum()) < 16:
            raise RuntimeError("too few valid optical-flow correspondences")

        sampled_backward = cv2.remap(
            backward, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        warped_second = cv2.remap(
            second, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        cycle = np.linalg.norm(forward + sampled_backward, axis=-1) / image_diagonal
        photometric = (
            np.abs(first.astype(np.float32) - warped_second.astype(np.float32))
            / 255.0
        )

        flow_gradients = [
            cv2.Sobel(forward[..., component], cv2.CV_32F, dx, 1 - dx, ksize=3)
            / 8.0
            for component in range(2)
            for dx in range(2)
        ]
        strain = np.sqrt(sum(gradient * gradient for gradient in flow_gradients))

        cycle_q90 = float(np.quantile(cycle[valid], 0.90))
        photometric_mean = float(photometric[valid].mean())
        strain_q95 = float(np.quantile(strain[valid], 0.95))

        # Fixed dimensionless reference scales.  They calibrate unlike units;
        # GRPO still performs the candidate-group normalization afterwards.
        pair_error = (
            cycle_q90 / 1.0e-3
            + photometric_mean / 2.0e-2
            + strain_q95 / 3.0e-1
        ) / 3.0
        pair_errors.append(pair_error)
        pair_diagnostics.append(
            {
                "cycle_q90": cycle_q90,
                "photometric_mean": photometric_mean,
                "strain_q95": strain_q95,
                "combined_error": pair_error,
            }
        )

    errors = torch.tensor(pair_errors, dtype=torch.float32, device=video.device)
    worst_k = min(topk_worst_pairs, errors.numel())
    worst_errors = torch.topk(errors, k=worst_k, largest=True).values
    diagnostics = {
        "resized_shape": (target_height, target_width),
        "pair_errors": pair_errors,
        "worst_pair_errors": worst_errors.detach().cpu().tolist(),
        "pairs": pair_diagnostics,
    }
    return worst_errors.mean(), diagnostics
