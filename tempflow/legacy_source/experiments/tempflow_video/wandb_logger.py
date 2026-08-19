"""Rank-zero W&B logging for standalone TempFlow runs.

The training path must remain usable without network access, so all W&B
failures are deliberately contained to this optional observer.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any


def _flatten(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for key, item in value.items():
            result.update(_flatten(item, f"{prefix}/{key}" if prefix else str(key)))
        return result
    if isinstance(value, bool):
        return {prefix: float(value)}
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return {prefix: float(value)}
    return {}


class TempFlowWandbLogger:
    """W&B observer with the same credentials and fallback behavior as AWM-CoCA."""

    def __init__(self, config: dict[str, Any], run_dir: str | Path, *, enabled: bool) -> None:
        self.run = None
        self.wandb = None
        self._init_thread: threading.Thread | None = None
        self._init_error: Exception | None = None
        self.run_dir = Path(run_dir)
        self.status_path = self.run_dir / "logs" / "wandb_run.txt"
        mode = str(config.get("logging", {}).get("wandb_mode", "offline")).lower()
        if not enabled or mode == "disabled":
            return
        # W&B login/init can block on DNS or the API.  Never hold the
        # distributed startup barrier while doing network I/O: rank 0 starts
        # the observer in the background and the training collectives proceed.
        self._init_thread = threading.Thread(
            target=self._initialize, args=(config, mode), daemon=True, name="wandb-init"
        )
        self._init_thread.start()

    def _initialize(self, config: dict[str, Any], mode: str) -> None:
        try:
            import wandb
            from experiments.awm_coca.wandb_monitor import (
                _BUNDLED_WANDB_API_KEY,
                _resolve_wandb_entity,
            )

            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            self.wandb = wandb
            if mode == "online":
                if wandb.login(
                    key=os.environ.get("WANDB_API_KEY") or _BUNDLED_WANDB_API_KEY,
                    relogin=True,
                ) is False:
                    raise RuntimeError("wandb.login returned False")
            kwargs: dict[str, Any] = {
                "project": os.environ.get("WANDB_PROJECT", "awm-coca"),
                "entity": _resolve_wandb_entity(),
                "name": os.environ.get("WANDB_NAME") or None,
                "dir": str(self.status_path.parent),
                "config": config,
                "mode": mode,
                "resume": "allow",
                "settings": wandb.Settings(
                    init_timeout=max(int(os.environ.get("WANDB_INIT_TIMEOUT", "15")), 1)
                ),
            }
            if os.environ.get("WANDB_RUN_ID"):
                kwargs["id"] = os.environ["WANDB_RUN_ID"]
            self.run = wandb.init(**kwargs)
            if self.run is None:
                raise RuntimeError("wandb.init returned None")
            self.run.define_metric("trainer/optimizer_step")
            self.run.define_metric("*", step_metric="trainer/optimizer_step")
            self.status_path.write_text(
                "status=active\n"
                f"mode={mode}\nproject={kwargs['project']}\n"
                f"run_id={getattr(self.run, 'id', '')}\n"
                f"url={getattr(self.run, 'url', 'offline')}\n",
                encoding="utf-8",
            )
            print(f"[INFO] TempFlow W&B enabled: {getattr(self.run, 'url', 'offline')}", flush=True)
        except Exception as exc:
            self._init_error = exc
            self._disable("initialization", exc)

    def _disable(self, stage: str, exc: Exception) -> None:
        print(f"[WARN] TempFlow W&B {stage} failed; retaining local logs: {exc}", flush=True)
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            self.status_path.write_text(
                f"status=fallback_local\nstage={stage}\nerror={type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        self.run = None

    def log_group(self, row: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            payload = {f"group/{key}": value for key, value in _flatten(row).items()}
            payload["trainer/optimizer_step"] = int(row.get("optimizer_step", 0))
            self.run.log(payload)
        except Exception as exc:
            self._disable("group upload", exc)

    def log_step(self, row: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            payload = {f"train/{key}": value for key, value in _flatten(row).items()}
            payload["trainer/optimizer_step"] = int(row["optimizer_step"])
            self.run.log(payload)
        except Exception as exc:
            self._disable("step upload", exc)

    def log_evaluation(self, row: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            payload = {f"eval/{key}": value for key, value in _flatten(row).items()}
            payload["trainer/optimizer_step"] = int(row.get("optimizer_step", 0))
            self.run.log(payload)
        except Exception as exc:
            self._disable("evaluation upload", exc)

    def finish(self) -> None:
        if self._init_thread is not None and self._init_thread.is_alive():
            self._init_thread.join(timeout=20.0)
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as exc:
            self._disable("finish", exc)
        finally:
            self.run = None
