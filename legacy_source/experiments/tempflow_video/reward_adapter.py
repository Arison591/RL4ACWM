from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from experiments.awm_coca.reward_runner import compute_head_reward
from experiments.tempflow_video.schemas import BranchRollout, OrdinaryRollout


def canonical_reward_config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class VideoRewardAdapter:
    """Thin, no-grad adapter around the legacy terminal video reward entry point."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.evaluator = str(self.config.get("evaluator", "legacy"))
        if self.evaluator not in {"legacy", "da3_mono"}:
            raise ValueError(f"unknown reward.evaluator={self.evaluator!r}")
        protocol = self.config.get(
            "time_alignment_protocol", "legacy_frame_index_truncate"
        )
        if protocol != "legacy_frame_index_truncate":
            raise ValueError(
                "the unchanged legacy reward only supports "
                f"time_alignment_protocol=legacy_frame_index_truncate, got {protocol!r}"
            )
        self.config_sha256 = canonical_reward_config_sha256(self.config)
        self._da3 = None
        if self.evaluator == "da3_mono":
            from experiments.tempflow_video.da3_video_reward import DA3MonoVideoReward

            self._da3 = DA3MonoVideoReward(self.config)

    def legacy_kwargs(
        self,
        *,
        condition_id: str,
        prep_dir: str,
        prediction_dir: str | Path,
    ) -> tuple[str, str, dict[str, Any]]:
        prediction_dir = Path(prediction_dir)
        template = self.config["gt_video_template"]
        gt_path = template.format(condition_id=condition_id)
        cameras = list(self.config.get("geometry_cameras", ("head", "hand_left", "hand_right")))
        templates = self.config.get("gt_video_templates", {})
        all_camera_videos = {
            camera: {
                "gt": templates.get(camera, template).format(
                    condition_id=condition_id, camera=camera
                ),
                "pred": str(prediction_dir / f"{camera}_color.mp4"),
            }
            for camera in cameras
        }
        kwargs = {
            "prep_dir": prep_dir,
            "max_frames": int(self.config.get("max_frames", 29)),
            "prompt": self.config.get("segmentation_prompt", "robot arm"),
            "confidence": float(self.config.get("confidence", 0.1)),
            "action_metric_weights": self.config.get("action_metric_weights"),
            "af_fdce_ate_norm_scale": float(self.config.get("af_fdce_ate_norm_scale", 0.2)),
            "fdce_scale": float(self.config.get("fdce_scale", 10.0)),
            "fdce_k": int(self.config.get("fdce_k", 16)),
            "fdce_visibility_threshold": float(
                self.config.get("fdce_visibility_threshold", 0.5)
            ),
            "fdce_min_visible_fraction": float(
                self.config.get("fdce_min_visible_fraction", 0.8)
            ),
            "fdce_min_common_frames": int(self.config.get("fdce_min_common_frames", 1)),
            "fdce_seed": int(self.config.get("fdce_seed", 0)),
            "reward_mode": self.config.get("mode", "action"),
            "geometry_enabled": bool(self.config.get("geometry_enabled", False)),
            "all_camera_videos": all_camera_videos,
            "geometry_cameras": cameras,
            "geometry_future_start": int(self.config.get("geometry_future_start", 4)),
            "geometry_future_end": int(self.config.get("geometry_future_end", 28)),
            "geometry_mean_weight": float(self.config.get("geometry_mean_weight", 0.6)),
            "geometry_worst_weight": float(self.config.get("geometry_worst_weight", 0.4)),
            "geometry_psnr_center_db": float(
                self.config.get("geometry_psnr_center_db", 20.4)
            ),
            "geometry_psnr_temperature_db": float(
                self.config.get("geometry_psnr_temperature_db", 1.8)
            ),
            "action_weight": float(self.config.get("action_weight", 1.0)),
            "geometry_weight": float(self.config.get("geometry_weight", 1.0)),
        }
        return gt_path, str(prediction_dir / "head_color.mp4"), kwargs

    @torch.no_grad()
    def score_paths(
        self,
        *,
        condition_id: str,
        prep_dir: str,
        prediction_dir: str | Path,
    ) -> dict[str, Any]:
        if self._da3 is not None:
            return self._da3.score_paths(
                condition_id=condition_id,
                prediction_dir=prediction_dir,
            )
        gt_path, pred_path, kwargs = self.legacy_kwargs(
            condition_id=condition_id,
            prep_dir=prep_dir,
            prediction_dir=prediction_dir,
        )
        return compute_head_reward(gt_path, pred_path, **kwargs)

    @torch.no_grad()
    def score_rollout(
        self, rollout: BranchRollout | OrdinaryRollout, *, prep_dir: str
    ) -> dict[str, Any]:
        reward = self.score_paths(
            condition_id=rollout.group_key.condition_id,
            prep_dir=prep_dir,
            prediction_dir=rollout.seed_dir,
        )
        rollout.reward = reward
        (rollout.seed_dir / "reward.json").write_text(
            json.dumps(reward, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return reward

    def load_models_for_preflight(self) -> dict[str, Any]:
        if self._da3 is None:
            return {"evaluator": self.evaluator, "model_loaded": False}
        return {
            "evaluator": self.evaluator,
            "model_loaded": True,
            "model": self._da3.load_model_for_preflight(),
        }
