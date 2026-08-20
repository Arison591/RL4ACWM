"""DA3 GT-guided pseudo-depth reward without camera calibration.

The reward deliberately compares a generated future clip with the matching GT
future clip after both have gone through the same frozen Depth Anything 3
pipeline.  It is not a metric-depth or reprojection reward.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


Representation = Literal["depth", "inverse", "log"]
Alignment = Literal["affine", "scale", "mad"]
AlignmentScope = Literal["view_clip", "global_clip"]
LossName = Literal["l1", "huber", "charbonnier"]
ConfidenceMode = Literal["none", "soft", "threshold"]


@dataclass(frozen=True)
class DA3RewardConfig:
    """Configuration shared by offline audit and RL-time reward calls.

    Alignment is intentionally fit over an entire future clip (and optionally
    all views), never independently per frame.  This retains temporal depth
    errors that per-frame scale/shift alignment would otherwise erase.
    """

    representation: Representation = "inverse"
    alignment: Alignment = "affine"
    alignment_scope: AlignmentScope = "view_clip"
    loss: LossName = "huber"
    trim_fraction: float = 0.05
    huber_delta_scale: float = 0.10
    charbonnier_eps: float = 1.0e-3
    confidence_mode: ConfidenceMode = "soft"
    confidence_floor: float = 0.05
    confidence_cap: float = 4.0
    confidence_threshold_quantile: float = 0.10
    border_fraction: float = 0.02
    dynamic_weight: float = 0.35
    max_alignment_samples: int = 100_000
    max_reduction_samples: int = 16_384
    eps: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.representation not in {"depth", "inverse", "log"}:
            raise ValueError(f"unsupported representation: {self.representation}")
        if self.alignment not in {"affine", "scale", "mad"}:
            raise ValueError(f"unsupported alignment: {self.alignment}")
        if self.alignment_scope not in {"view_clip", "global_clip"}:
            raise ValueError(f"unsupported alignment scope: {self.alignment_scope}")
        if self.loss not in {"l1", "huber", "charbonnier"}:
            raise ValueError(f"unsupported loss: {self.loss}")
        if self.confidence_mode not in {"none", "soft", "threshold"}:
            raise ValueError(f"unsupported confidence mode: {self.confidence_mode}")
        if not 0.0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        if not 0.0 <= self.border_fraction < 0.5:
            raise ValueError("border_fraction must be in [0, 0.5)")
        if not 0.0 <= self.dynamic_weight <= 1.0:
            raise ValueError("dynamic_weight must be in [0, 1]")
        if self.max_alignment_samples < 16:
            raise ValueError("max_alignment_samples must be at least 16")
        if self.max_reduction_samples < 256:
            raise ValueError("max_reduction_samples must be at least 256")


@dataclass
class DA3RewardResult:
    """Scalar reward plus audit values for one candidate clip."""

    error: float
    reward: float
    full_error: float
    dynamic_error: float
    per_view_error: Tensor
    per_view_full_error: Tensor
    per_view_dynamic_error: Tensor
    per_view_frame_error: Tensor
    valid_pixel_ratio: Tensor
    confident_pixel_ratio: Tensor
    alignment_scale: Tensor
    alignment_shift: Tensor
    dynamic_pixel_ratio: Tensor

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": self.error,
            "reward": self.reward,
            "full_error": self.full_error,
            "dynamic_error": self.dynamic_error,
            "per_view_error": self.per_view_error.detach().cpu().tolist(),
            "per_view_full_error": self.per_view_full_error.detach().cpu().tolist(),
            "per_view_dynamic_error": self.per_view_dynamic_error.detach().cpu().tolist(),
            "per_view_frame_error": self.per_view_frame_error.detach().cpu().tolist(),
            "valid_pixel_ratio": self.valid_pixel_ratio.detach().cpu().tolist(),
            "confident_pixel_ratio": self.confident_pixel_ratio.detach().cpu().tolist(),
            "alignment_scale": self.alignment_scale.detach().cpu().tolist(),
            "alignment_shift": self.alignment_shift.detach().cpu().tolist(),
            "dynamic_pixel_ratio": self.dynamic_pixel_ratio.detach().cpu().tolist(),
        }


def _as_view_time(value: Tensor, name: str) -> Tensor:
    value = torch.as_tensor(value)
    if value.ndim == 3:
        return value.unsqueeze(0)
    if value.ndim == 4:
        return value
    raise ValueError(f"{name} must be [T,H,W] or [V,T,H,W], got {tuple(value.shape)}")


def _to_representation(depth: Tensor, config: DA3RewardConfig) -> Tensor:
    positive = depth.clamp_min(config.eps)
    if config.representation == "depth":
        return positive
    if config.representation == "inverse":
        return positive.reciprocal()
    return positive.log()


def _border_valid_mask(shape: Sequence[int], border_fraction: float, device: torch.device) -> Tensor:
    _, _, height, width = shape
    mask = torch.ones(shape, dtype=torch.bool, device=device)
    margin_h = int(round(height * border_fraction))
    margin_w = int(round(width * border_fraction))
    if margin_h:
        mask[..., :margin_h, :] = False
        mask[..., height - margin_h :, :] = False
    if margin_w:
        mask[..., :, :margin_w] = False
        mask[..., :, width - margin_w :] = False
    return mask


def _sample_pair(source: Tensor, target: Tensor, valid: Tensor, max_samples: int) -> tuple[Tensor, Tensor]:
    source_values = source[valid].float()
    target_values = target[valid].float()
    if source_values.numel() > max_samples:
        indices = torch.linspace(
            0, source_values.numel() - 1, max_samples, device=source_values.device
        ).round().long()
        source_values = source_values[indices]
        target_values = target_values[indices]
    return source_values, target_values


def _fit_alignment(
    generated: Tensor,
    reference: Tensor,
    valid: Tensor,
    config: DA3RewardConfig,
) -> tuple[Tensor, Tensor]:
    source, target = _sample_pair(generated, reference, valid, config.max_alignment_samples)
    if source.numel() < 16:
        return generated.new_tensor(1.0), generated.new_tensor(0.0)

    if config.alignment == "mad":
        source_median = source.median()
        target_median = target.median()
        source_mad = (source - source_median).abs().median().clamp_min(config.eps)
        target_mad = (target - target_median).abs().median().clamp_min(config.eps)
        return target_mad / source_mad, target_median - source_median * target_mad / source_mad

    if config.alignment == "scale":
        denominator = source.square().mean().clamp_min(config.eps)
        return (source * target).mean() / denominator, generated.new_tensor(0.0)

    source_mean = source.mean()
    target_mean = target.mean()
    centered_source = source - source_mean
    scale = (centered_source * (target - target_mean)).mean() / centered_source.square().mean().clamp_min(config.eps)
    return scale, target_mean - scale * source_mean


def _confidence_weights(
    gt_confidence: Optional[Tensor],
    gen_confidence: Optional[Tensor],
    valid: Tensor,
    config: DA3RewardConfig,
) -> tuple[Tensor, Tensor]:
    ones = torch.ones(valid.shape, dtype=torch.float32, device=valid.device)
    if config.confidence_mode == "none" or gt_confidence is None or gen_confidence is None:
        return ones, valid

    gt = gt_confidence.float().clamp_min(0.0)
    generated = gen_confidence.float().clamp_min(0.0)
    reference_values = gt[valid]
    generated_values = generated[valid]
    if reference_values.numel() < 16 or generated_values.numel() < 16:
        return ones, valid

    if config.confidence_mode == "threshold":
        threshold_gt = torch.quantile(reference_values, config.confidence_threshold_quantile)
        threshold_gen = torch.quantile(generated_values, config.confidence_threshold_quantile)
        confident = valid & (gt >= threshold_gt) & (generated >= threshold_gen)
        return ones, confident

    gt_median = reference_values.median().clamp_min(config.eps)
    gen_median = generated_values.median().clamp_min(config.eps)
    gt_normalized = (gt / gt_median).clamp(config.confidence_floor, config.confidence_cap)
    gen_normalized = (generated / gen_median).clamp(config.confidence_floor, config.confidence_cap)
    return (gt_normalized * gen_normalized).sqrt(), valid


def _loss_from_difference(difference: Tensor, target: Tensor, config: DA3RewardConfig) -> Tensor:
    absolute = difference.abs()
    if config.loss == "l1":
        return absolute
    if config.loss == "charbonnier":
        return torch.sqrt(difference.square() + config.charbonnier_eps**2)
    scale = (target - target.median()).abs().median().clamp_min(config.eps)
    delta = config.huber_delta_scale * scale
    return torch.where(absolute <= delta, 0.5 * difference.square() / delta, absolute - 0.5 * delta)


def _trimmed_weighted_mean(
    values: Tensor,
    weights: Tensor,
    valid: Tensor,
    trim_fraction: float,
    max_samples: int,
) -> Tensor:
    values = values[valid]
    weights = weights[valid]
    if values.numel() == 0:
        return values.new_tensor(float("nan"))
    if values.numel() > max_samples:
        indices = torch.linspace(0, values.numel() - 1, max_samples, device=values.device).round().long()
        values, weights = values[indices], weights[indices]
    if trim_fraction:
        keep = max(1, int(math.floor(values.numel() * (1.0 - trim_fraction))))
        indices = torch.argsort(values)[:keep]
        values, weights = values[indices], weights[indices]
    return (values * weights).sum() / weights.sum().clamp_min(1.0e-12)


def _error_for_mask(
    loss: Tensor,
    weights: Tensor,
    valid: Tensor,
    dynamic_mask: Optional[Tensor],
    trim_fraction: float,
    max_samples: int,
) -> tuple[Tensor, Tensor]:
    full = _trimmed_weighted_mean(loss, weights, valid, trim_fraction, max_samples)
    if dynamic_mask is None:
        return full, full
    dynamic_valid = valid & dynamic_mask.bool()
    if dynamic_valid.sum() < 16:
        return full, full
    return full, _trimmed_weighted_mean(loss, weights, dynamic_valid, trim_fraction, max_samples)


def make_gt_motion_mask(
    gt_rgb: Tensor,
    observed_last_rgb: Tensor,
    quantile: float = 0.80,
    dilation_radius: int = 5,
) -> Tensor:
    """Build a reproducible GT-only change mask for [V,T,3,H,W] RGB clips.

    It intentionally uses no generated pixels or ranking information.  The
    light dilation makes a change mask cover the moving gripper/object and its
    immediate contact neighbourhood rather than isolated difference pixels.
    """

    clip = torch.as_tensor(gt_rgb).float()
    observed = torch.as_tensor(observed_last_rgb).float()
    if clip.ndim == 4:
        clip = clip.unsqueeze(0)
    if observed.ndim == 3:
        observed = observed.unsqueeze(0)
    if clip.ndim != 5 or observed.ndim != 4:
        raise ValueError("gt_rgb must be [V,T,3,H,W] and observed_last_rgb [V,3,H,W]")
    if clip.shape[0] != observed.shape[0] or clip.shape[2:] != observed.shape[1:]:
        raise ValueError("GT future and observed-last RGB shapes are incompatible")
    change = (clip - observed[:, None]).abs().mean(dim=2)
    thresholds = torch.quantile(change.flatten(2), quantile, dim=-1, keepdim=True)
    mask = change >= thresholds[..., None]
    if dilation_radius:
        kernel = dilation_radius * 2 + 1
        mask = F.max_pool2d(mask.float().flatten(0, 1).unsqueeze(1), kernel, stride=1, padding=dilation_radius)
        mask = mask.squeeze(1).reshape_as(change).bool()
    return mask


def compute_da3_gt_reward(
    gt_depth: Tensor,
    gen_depth: Tensor,
    gt_confidence: Optional[Tensor] = None,
    gen_confidence: Optional[Tensor] = None,
    dynamic_mask: Optional[Tensor] = None,
    config: DA3RewardConfig = DA3RewardConfig(),
    include_per_frame: bool = True,
) -> DA3RewardResult:
    """Compare matching DA3 pseudo-depth clips with shared clip alignment.

    Args are [V,T,H,W] or [T,H,W].  Output is an error (lower is better) and
    its sign-flipped reward.  Neither camera calibration nor RGB similarity is
    used in the score.
    """

    gt = _as_view_time(gt_depth, "gt_depth").float()
    generated = _as_view_time(gen_depth, "gen_depth").to(gt.device).float()
    if gt.shape != generated.shape:
        raise ValueError(f"depth shapes must match, got {tuple(gt.shape)} and {tuple(generated.shape)}")
    gt_conf = _as_view_time(gt_confidence, "gt_confidence").to(gt.device).float() if gt_confidence is not None else None
    gen_conf = _as_view_time(gen_confidence, "gen_confidence").to(gt.device).float() if gen_confidence is not None else None
    if gt_conf is not None and gt_conf.shape != gt.shape:
        raise ValueError("gt_confidence shape must match gt_depth")
    if gen_conf is not None and gen_conf.shape != gt.shape:
        raise ValueError("gen_confidence shape must match gen_depth")
    dynamic = _as_view_time(dynamic_mask, "dynamic_mask").to(gt.device).bool() if dynamic_mask is not None else None
    if dynamic is not None and dynamic.shape != gt.shape:
        raise ValueError("dynamic_mask shape must match depth tensors")

    gt_q, gen_q = _to_representation(gt, config), _to_representation(generated, config)
    finite = torch.isfinite(gt_q) & torch.isfinite(gen_q) & (gt > config.eps) & (generated > config.eps)
    valid = finite & _border_valid_mask(gt.shape, config.border_fraction, gt.device)
    weights, confidence_valid = _confidence_weights(gt_conf, gen_conf, valid, config)
    valid &= confidence_valid

    views, frames = gt.shape[:2]
    scales = torch.empty(views, dtype=gt.dtype, device=gt.device)
    shifts = torch.empty_like(scales)
    aligned = torch.empty_like(gen_q)
    if config.alignment_scope == "global_clip":
        scale, shift = _fit_alignment(gen_q, gt_q, valid, config)
        scales.fill_(scale)
        shifts.fill_(shift)
        aligned.copy_(gen_q * scale + shift)
    else:
        for view in range(views):
            scale, shift = _fit_alignment(gen_q[view], gt_q[view], valid[view], config)
            scales[view], shifts[view] = scale, shift
            aligned[view] = gen_q[view] * scale + shift

    loss = _loss_from_difference(aligned - gt_q, gt_q, config)
    per_view_full, per_view_dynamic, per_frame = [], [], []
    for view in range(views):
        full, dynamic_error = _error_for_mask(loss[view], weights[view], valid[view], None if dynamic is None else dynamic[view], config.trim_fraction, config.max_reduction_samples)
        per_view_full.append(full)
        per_view_dynamic.append(dynamic_error)
        if include_per_frame:
            per_frame.append(torch.stack([
                _error_for_mask(loss[view, frame], weights[view, frame], valid[view, frame], None if dynamic is None else dynamic[view, frame], config.trim_fraction, config.max_reduction_samples)[0]
                for frame in range(frames)
            ]))
    per_view_full_t = torch.stack(per_view_full)
    per_view_dynamic_t = torch.stack(per_view_dynamic)
    per_view_error = (1.0 - config.dynamic_weight) * per_view_full_t + config.dynamic_weight * per_view_dynamic_t
    full_error = per_view_full_t.mean()
    dynamic_error = per_view_dynamic_t.mean()
    error = per_view_error.mean()
    dynamic_ratio = dynamic.float().mean(dim=(1, 2, 3)) if dynamic is not None else torch.zeros(views, device=gt.device)
    return DA3RewardResult(
        error=float(error.detach().cpu()),
        reward=float((-error).detach().cpu()),
        full_error=float(full_error.detach().cpu()),
        dynamic_error=float(dynamic_error.detach().cpu()),
        per_view_error=per_view_error,
        per_view_full_error=per_view_full_t,
        per_view_dynamic_error=per_view_dynamic_t,
        per_view_frame_error=torch.stack(per_frame) if include_per_frame else loss.new_empty((views, 0)),
        valid_pixel_ratio=valid.float().mean(dim=(1, 2, 3)),
        confident_pixel_ratio=confidence_valid.float().mean(dim=(1, 2, 3)),
        alignment_scale=scales,
        alignment_shift=shifts,
        dynamic_pixel_ratio=dynamic_ratio,
    )


def median_mad_advantage(reward: Tensor, eps: float = 1.0e-6, clip: Optional[float] = 5.0) -> Tensor:
    """Group-normalize raw rewards for GRPO without rescaling their meaning."""

    values = torch.as_tensor(reward).float()
    if values.ndim not in {1, 2}:
        raise ValueError("reward must be [K] or [B,K]")
    median = values.median(dim=-1, keepdim=True).values
    mad = (values - median).abs().median(dim=-1, keepdim=True).values
    advantage = (values - median) / (1.4826 * mad + eps)
    return advantage.clamp(-clip, clip) if clip is not None else advantage


def save_da3_cache(
    path: str | Path,
    depth: Tensor,
    confidence: Optional[Tensor],
    metadata: Mapping[str, Any],
) -> None:
    """Write a versioned depth/confidence cache without serializing a model."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "depth": torch.as_tensor(depth).detach().cpu().float(),
        "confidence": None if confidence is None else torch.as_tensor(confidence).detach().cpu().float(),
        "metadata": dict(metadata),
    }
    torch.save(payload, target)


def load_da3_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or "depth" not in payload:
        raise ValueError(f"unsupported DA3 cache: {path}")
    depth = torch.as_tensor(payload["depth"]).float()
    confidence = payload.get("confidence")
    if depth.ndim not in {3, 4} or not torch.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError(f"invalid pseudo-depth cache: {path}")
    if confidence is not None:
        confidence = torch.as_tensor(confidence).float()
        if confidence.shape != depth.shape or not torch.isfinite(confidence).all():
            raise ValueError(f"invalid confidence cache: {path}")
    return {"depth": depth, "confidence": confidence, "metadata": payload.get("metadata", {})}


def _source_provenance(source_root: Path) -> dict[str, Any]:
    """Return source revision information without making git a runtime dependency."""

    repo = source_root.parent
    result: dict[str, Any] = {"source_root": str(source_root)}
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return result
    result.update({"source_commit": revision, "source_dirty": bool(status)})
    return result


class DA3ModelRunner:
    """Frozen DA3 runner with explicitly separate mono and joint modes."""

    def __init__(
        self,
        model_id: str = "depth-anything/DA3-BASE",
        device: str | torch.device = "cuda",
        process_res: int = 336,
        process_res_method: str = "upper_bound_resize",
        source_root: str | Path = "/home/ma-user/modelarts/user-job-dir/external/Depth-Anything-3/src",
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        self.process_res = int(process_res)
        self.process_res_method = process_res_method
        self.source_root = Path(source_root)
        self.model: Any = None

    def _load(self) -> Any:
        if self.model is not None:
            return self.model
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as error:
            raise ImportError(
                "Depth Anything 3 is unavailable. Install the official source and its inference dependencies, "
                f"or pass source_root to a valid checkout. Original error: {error}"
            ) from error
        # The audited checkpoint is already cached locally; do not make reward
        # execution depend on an unnecessary Hub metadata request.
        self.model = DepthAnything3.from_pretrained(self.model_id, local_files_only=True).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        return self.model

    def _preprocess(self, images: Sequence[np.ndarray | str]) -> Tensor:
        model = self._load()
        processed, _, _ = model._preprocess_inputs(
            list(images), None, None, self.process_res, self.process_res_method
        )
        return processed

    def _forward(self, processed: Tensor, views_per_item: int) -> tuple[Tensor, Optional[Tensor]]:
        if processed.ndim != 4 or processed.shape[0] % views_per_item:
            raise ValueError("processed DA3 input has an invalid batch/view layout")
        model = self._load()
        batch = processed.reshape(-1, views_per_item, *processed.shape[1:]).to(self.device, non_blocking=True).float()
        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=autocast_dtype, enabled=self.device.type == "cuda"):
            raw = model.model(batch, None, None, [], False, False, "first")
        depth = raw["depth"]
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        elif depth.ndim == 5 and depth.shape[2] == 1:
            depth = depth.squeeze(2)
        if depth.ndim != 4:
            raise RuntimeError(f"unexpected DA3 depth shape {tuple(depth.shape)}")
        confidence = raw.get("depth_conf")
        if confidence is not None:
            if confidence.ndim == 5 and confidence.shape[-1] == 1:
                confidence = confidence.squeeze(-1)
            elif confidence.ndim == 5 and confidence.shape[2] == 1:
                confidence = confidence.squeeze(2)
            if confidence.shape != depth.shape:
                raise RuntimeError(f"unexpected DA3 confidence shape {tuple(confidence.shape)}")
        return depth.detach().float().cpu(), None if confidence is None else confidence.detach().float().cpu()

    def infer_mono(self, images: Sequence[np.ndarray | str]) -> tuple[Tensor, Optional[Tensor]]:
        """Infer each image independently, batched only along B and never N."""

        processed = self._preprocess(images)
        depth, confidence = self._forward(processed, views_per_item=1)
        return depth[:, 0], None if confidence is None else confidence[:, 0]

    def infer_joint(self, view_images: Sequence[Sequence[np.ndarray | str]]) -> tuple[Tensor, Optional[Tensor]]:
        """Infer synchronized frames jointly: B=time, N=view, with no poses."""

        if len(view_images) < 2:
            raise ValueError("joint DA3 inference requires at least two views")
        frame_count = len(view_images[0])
        if frame_count < 1 or any(len(view) != frame_count for view in view_images):
            raise ValueError("all joint views must have the same non-zero frame count")
        flattened = [view_images[view][frame] for frame in range(frame_count) for view in range(len(view_images))]
        processed = self._preprocess(flattened)
        depth, confidence = self._forward(processed, views_per_item=len(view_images))
        return depth.permute(1, 0, 2, 3).contiguous(), None if confidence is None else confidence.permute(1, 0, 2, 3).contiguous()

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "backend": "depth-anything-3",
            "model_id": self.model_id,
            "process_res": self.process_res,
            "process_res_method": self.process_res_method,
            "device": str(self.device),
            "torch_version": getattr(torch, "__version__", "unknown"),
            **_source_provenance(self.source_root),
        }
        if self.model is not None:
            metadata["model_revision"] = getattr(self.model, "_commit_hash", None)
        return metadata


__all__ = [
    "DA3ModelRunner",
    "DA3RewardConfig",
    "DA3RewardResult",
    "compute_da3_gt_reward",
    "load_da3_cache",
    "make_gt_motion_mask",
    "median_mad_advantage",
    "save_da3_cache",
]
