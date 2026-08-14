from __future__ import annotations

import torch


def _as_multiview(value: torch.Tensor, n_view: int) -> torch.Tensor:
    if value.ndim == 6:
        return value
    if value.ndim != 5 or value.shape[0] % n_view:
        raise ValueError(f"expected [B*V,C,T,H,W] or [B,V,C,T,H,W], got {tuple(value.shape)}")
    return value.reshape(value.shape[0] // n_view, n_view, *value.shape[1:])


def masked_video_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    n_view: int,
    valid_future_mask: torch.Tensor | None = None,
    camera_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    prediction = _as_multiview(prediction, n_view)
    target = _as_multiview(target, n_view)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    error = (prediction.float() - target.float()).square().mean(dim=(2, 4, 5))
    batch, views, frames = error.shape
    if valid_future_mask is None:
        mask = torch.ones((batch, views, frames), device=error.device, dtype=error.dtype)
    else:
        mask = valid_future_mask.to(device=error.device, dtype=error.dtype)
        if mask.ndim == 1:
            mask = mask[None, None, :].expand(batch, views, -1)
        if mask.ndim == 2:
            mask = mask[:, None, :].expand(-1, views, -1)
        if mask.shape != error.shape:
            raise ValueError(f"invalid future mask shape: {mask.shape}, expected {error.shape}")
    per_view = (error * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
    if camera_weights is None:
        weights = torch.full((views,), 1.0 / views, device=error.device, dtype=error.dtype)
    else:
        weights = camera_weights.to(device=error.device, dtype=error.dtype)
        if weights.shape != (views,) or torch.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("camera_weights must be non-negative [V] with positive sum")
        weights = weights / weights.sum()
    return (per_view * weights).sum(-1).mean()
