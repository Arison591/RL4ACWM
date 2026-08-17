from __future__ import annotations

import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments.awm_coca.advantage import leave_one_out_advantages


# 远端训练交付使用的 W&B 凭据；环境变量 WANDB_API_KEY 仍可覆盖以便轮换。
_BUNDLED_WANDB_API_KEY = "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"
_BUNDLED_WANDB_ENTITY = "hrqian06-huazhong-university-of-science-and-technology"
_LEGACY_INVALID_WANDB_ENTITIES = {"hrqian06"}


def _resolve_wandb_entity() -> str:
    """Use the verified team slug and repair the legacy short user entity."""
    configured = os.environ.get("WANDB_ENTITY", "").strip()
    if not configured or configured in _LEGACY_INVALID_WANDB_ENTITIES:
        return _BUNDLED_WANDB_ENTITY
    return configured


def _finite(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _summary(prefix: str, values: Iterable[Any]) -> dict[str, float]:
    numbers = _finite(values)
    if not numbers:
        return {}
    return {
        f"{prefix}/mean": statistics.fmean(numbers),
        f"{prefix}/min": min(numbers),
        f"{prefix}/max": max(numbers),
        f"{prefix}/std": statistics.pstdev(numbers),
    }


def _advantage_diagnostics(
    rewards: Iterable[dict[str, Any]],
    *,
    action_weight: float,
    geometry_weight: float,
) -> dict[str, float]:
    """Diagnose per-seed competition between action and geometry LOO advantages."""
    paired = []
    for reward in rewards:
        if reward.get("total_reward") is None or not reward.get("valid", True):
            continue
        values = _finite((reward.get("action_reward"), reward.get("geometry_reward")))
        if len(values) == 2:
            paired.append((values[0], values[1]))
    if len(paired) < 2:
        return {}

    weight_sum = action_weight + geometry_weight
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        return {}
    action_weight /= weight_sum
    geometry_weight /= weight_sum

    _, action_advantages = leave_one_out_advantages([value[0] for value in paired])
    _, geometry_advantages = leave_one_out_advantages([value[1] for value in paired])

    action_std = statistics.pstdev(action_advantages)
    geometry_std = statistics.pstdev(geometry_advantages)
    if action_std > 0.0 and geometry_std > 0.0:
        correlation = statistics.fmean(
            (action / action_std) * (geometry / geometry_std)
            for action, geometry in zip(action_advantages, geometry_advantages)
        )
        correlation = max(-1.0, min(1.0, correlation))
    else:
        correlation = 0.0

    action_contributions = [action_weight * value for value in action_advantages]
    geometry_contributions = [geometry_weight * value for value in geometry_advantages]
    total_advantages = [
        action + geometry
        for action, geometry in zip(action_contributions, geometry_contributions)
    ]
    flip_rate = statistics.fmean(
        float(action * total < 0.0)
        for action, total in zip(action_advantages, total_advantages)
    )
    separate_magnitude = sum(
        abs(action) + abs(geometry)
        for action, geometry in zip(action_contributions, geometry_contributions)
    )
    retained_magnitude = sum(abs(value) for value in total_advantages)
    cancellation_rate = (
        1.0 - retained_magnitude / separate_magnitude
        if separate_magnitude > 0.0 else 0.0
    )

    return {
        "advantage/action_std": action_std,
        "advantage/geometry_std": geometry_std,
        "advantage/action_geometry_corr": correlation,
        "advantage/action_flip_rate": flip_rate,
        "advantage/cancellation_rate": max(0.0, min(1.0, cancellation_rate)),
    }


class WandbMonitor:
    """Rank-0 W&B monitor. Any W&B failure disables only remote logging."""

    def __init__(self, config: dict[str, Any], output_dir: str | Path, *, enabled: bool = True) -> None:
        self.run = None
        self.wandb = None
        self.output_dir = Path(output_dir)
        self.video_every = max(int(os.environ.get("WANDB_VIDEO_EVERY", "50")), 0)
        self.video_samples = max(int(os.environ.get("WANDB_VIDEO_SAMPLES", "1")), 0)
        self.status_path = self.output_dir / "logs" / "wandb_run.txt"
        reward_config = config.get("reward", {})
        reward_mode = str(reward_config.get("mode", "action")).lower()
        if reward_mode == "joint":
            self.action_reward_weight = float(reward_config.get("action_weight", 1.0))
            self.geometry_reward_weight = float(reward_config.get("geometry_weight", 1.0))
        elif reward_mode in {"geometry", "psnr"}:
            self.action_reward_weight = 0.0
            self.geometry_reward_weight = 1.0
        else:
            self.action_reward_weight = 1.0
            self.geometry_reward_weight = 0.0
        if not enabled:
            return
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        mode = os.environ.get("WANDB_MODE", "online").strip().lower()
        if mode == "disabled":
            self._write_status("status=disabled\n")
            print("[INFO] W&B 已关闭；本地 JSONL 和控制台日志仍正常记录。")
            return
        try:
            import wandb

            self.wandb = wandb
            api_key = os.environ.get("WANDB_API_KEY") or _BUNDLED_WANDB_API_KEY
            if mode == "online":
                logged_in = wandb.login(key=api_key, relogin=True)
                if logged_in is False:
                    raise RuntimeError("wandb.login returned False")
            init_kwargs: dict[str, Any] = {
                "project": os.environ.get("WANDB_PROJECT", "awm-coca"),
                "name": os.environ.get("WANDB_NAME") or None,
                "dir": str(self.output_dir / "logs"),
                "config": config,
                "mode": mode,
                "resume": "allow",
                "settings": wandb.Settings(
                    init_timeout=max(int(os.environ.get("WANDB_INIT_TIMEOUT", "15")), 1),
                ),
            }
            entity = _resolve_wandb_entity()
            run_id = os.environ.get("WANDB_RUN_ID")
            if entity:
                init_kwargs["entity"] = entity
            if run_id:
                init_kwargs["id"] = run_id
            self.run = wandb.init(**init_kwargs)
            if self.run is None:
                raise RuntimeError("wandb.init returned None")
            self.run.define_metric("trainer/group_step")
            self.run.define_metric("*", step_metric="trainer/group_step")
            url = getattr(self.run, "url", None) or "offline"
            self._write_status(
                f"status=active\nmode={mode}\nproject={init_kwargs['project']}\n"
                f"run_id={getattr(self.run, 'id', '')}\nurl={url}\n"
            )
            print(f"[INFO] W&B 已启用: {url}")
        except Exception as exc:
            self._disable("初始化失败", exc)

    def _write_status(self, text: str) -> None:
        self.status_path.write_text(text, encoding="utf-8")

    def _disable(self, stage: str, exc: Exception) -> None:
        print(f"[WARN] W&B {stage}，已自动降级为本地日志: {type(exc).__name__}: {exc}")
        try:
            with self.status_path.open("a", encoding="utf-8") as handle:
                handle.write(f"status=fallback_local\nstage={stage}\nerror={type(exc).__name__}: {exc}\n")
        except OSError:
            pass
        self.run = None

    def _scalar_payload(
        self,
        train_row: dict[str, Any],
        rollout_rows: list[dict[str, Any]],
        *,
        group_step: int,
    ) -> dict[str, Any]:
        samples = train_row.get("samples", [])
        payload: dict[str, Any] = {
            "trainer/group_step": int(group_step),
            "trainer/optimizer_step": int(train_row["optimizer_step"]),
            "trainer/policy_version": int(train_row["policy_version"]),
            "trainer/loss": float(train_row["loss"]),
            "trainer/learning_rate": float(train_row["learning_rate"]),
            "trainer/world_size": int(train_row.get("world_size", 1)),
            "trainer/optimizer_stepped": int(bool(train_row.get("optimizer_stepped"))),
            "trainer/epoch": float(train_row.get("epoch", 0.0)),
            "trainer/epoch_index": int(train_row.get("epoch_index", 1)),
            "trainer/condition_position": int(train_row.get("condition_position", 0)),
            "trainer/conditions_per_epoch": int(train_row.get("conditions_per_epoch", 0)),
            "trainer/group_duration_seconds": float(train_row.get("group_duration_seconds", 0.0)),
        }
        if train_row.get("mean_reward") is not None:
            payload["trainer/mean_reward"] = float(train_row["mean_reward"])
        if train_row.get("grad_norm") is not None:
            payload["trainer/grad_norm"] = float(train_row["grad_norm"])
        for key in ("fm_grad_norm", "kl_grad_norm"):
            if train_row.get(key) is not None:
                payload[f"trainer/{key}"] = float(train_row[key])
        for key in (
            "advantage", "action_advantage", "geometry_advantage", "fm_loss", "reference_kl",
            "weighted_loss", "importance_weight",
            "proposal_probability", "noise_time", "noise_level_index",
        ):
            payload.update(_summary(f"train_samples/{key}", (row.get(key) for row in samples)))

        rewards = [row.get("reward", {}) for row in rollout_rows]
        payload.update(_summary("reward/total", (row.get("total_reward") for row in rewards)))
        payload.update(_summary("reward/action", (row.get("action_reward") for row in rewards)))
        payload.update(_summary("reward/geometry", (row.get("geometry_reward") for row in rewards)))
        payload.update(_advantage_diagnostics(
            rewards,
            action_weight=self.action_reward_weight,
            geometry_weight=self.geometry_reward_weight,
        ))
        payload["reward/valid_fraction"] = statistics.fmean(
            [float(bool(row.get("valid"))) for row in rewards]
        ) if rewards else 0.0

        action_keys = (
            "mean_iou", "ate_norm", "fdce", "af_fdce_ate_norm", "det_coverage",
            "af_fdce_det_coverage",
        )
        for key in action_keys:
            payload.update(_summary(f"action/{key}", (row.get("action_metrics", {}).get(key) for row in rewards)))
        payload.update(_summary(
            "geometry/balanced_psnr_db",
            (row.get("geometry", {}).get("metrics", {}).get("balanced_psnr_db") for row in rewards),
        ))
        payload.update(_summary(
            "geometry/mean_psnr_db",
            (row.get("geometry", {}).get("metrics", {}).get("mean_psnr_db") for row in rewards),
        ))
        for camera in ("head", "hand_left", "hand_right"):
            payload.update(_summary(
                f"geometry/psnr_{camera}_db",
                (
                    row.get("geometry", {}).get("metrics", {}).get("per_view_psnr_db", {}).get(camera)
                    for row in rewards
                ),
            ))
        payload.update(_summary(
            "credit/reward_conservation_error",
            (row.get("credit", {}).get("reward_conservation_error") for row in rollout_rows),
        ))
        return payload

    def _video_payload(
        self,
        rollout_rows: list[dict[str, Any]],
        *,
        optimizer_step: int,
    ) -> dict[str, Any]:
        if (
            self.wandb is None
            or self.video_every <= 0
            or self.video_samples <= 0
            or not (optimizer_step == 1 or optimizer_step % self.video_every == 0)
        ):
            return {}
        payload = {}
        for row in rollout_rows[: self.video_samples]:
            seed_dir = (
                self.output_dir / "rollouts" / str(row["group_id"]) / str(row["sample_id"])
            )
            for camera in ("head", "hand_left", "hand_right"):
                path = seed_dir / f"{camera}_color.mp4"
                if path.is_file():
                    key = f"rollout_video/{camera}_{row['sample_id']}"
                    payload[key] = self.wandb.Video(str(path), format="mp4")
        return payload

    def log_group(
        self,
        train_row: dict[str, Any],
        rollout_rows: list[dict[str, Any]],
        *,
        group_step: int,
    ) -> None:
        if self.run is None:
            return
        try:
            payload = self._scalar_payload(train_row, rollout_rows, group_step=group_step)
            payload.update(self._video_payload(
                rollout_rows, optimizer_step=int(train_row["optimizer_step"])
            ))
            self.run.log(payload)
        except Exception as exc:
            self._disable("上传失败", exc)

    def log_eval(self, payload: dict[str, Any], *, group_step: int) -> None:
        """Log a held-out evaluation round (keys are already eval/*-prefixed)."""
        if self.run is None:
            return
        try:
            self.run.log({"trainer/group_step": int(group_step), **payload})
        except Exception as exc:
            self._disable("上传失败", exc)

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as exc:
            self._disable("结束 run 失败", exc)
        finally:
            self.run = None

    def __enter__(self) -> "WandbMonitor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.finish()
