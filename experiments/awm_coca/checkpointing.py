from __future__ import annotations

import json
import os
import random
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _clone_state(value: Any) -> Any:
    """Deep-copy tensors to CPU / numpy arrays so a background thread can
    serialize a checkpoint snapshot while training keeps mutating live objects."""
    if torch.is_tensor(value):
        return value.detach().to("cpu").clone()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_state(item) for item in value)
    return value


def build_checkpoint_snapshot(
    *,
    step: int,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    ema_state: dict[str, Any],
    sampler_state: dict[str, Any],
    trainer_state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Capture all checkpoint state as CPU tensors *now*, so a background thread
    can write it to disk later without racing against the training loop."""
    if hasattr(policy, "get_peft_model_state_dict"):
        policy_lora = _clone_state(policy.get_peft_model_state_dict())
        peft_configs = getattr(policy, "peft_config", {})
        default_config = peft_configs.get("default") if isinstance(peft_configs, dict) else None
        adapter_config = default_config.to_dict() if default_config is not None else {}
    else:
        policy_lora = None
        adapter_config = {}
    return {
        "step": int(step),
        "policy_lora": policy_lora,
        "adapter_config": adapter_config,
        "optimizer": _clone_state(optimizer.state_dict()),
        "lr_scheduler": _clone_state(lr_scheduler.state_dict() if lr_scheduler is not None else {}),
        "ema": _clone_state(ema_state),
        "rng": capture_rng_state(),
        "sampler": _clone_state(sampler_state),
        "trainer_state": _clone_state(trainer_state),
        "config": config,
    }


def save_checkpoint_from_snapshot(output_dir: str | Path, *, step: int, snapshot: dict[str, Any]) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"checkpoint_{step}"
    if final.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")
    with tempfile.TemporaryDirectory(prefix=f"checkpoint_{step}_", dir=root) as temp_name:
        temp = Path(temp_name)
        policy_dir = temp / "policy_lora"
        policy_dir.mkdir()
        if snapshot.get("policy_lora"):
            from safetensors.torch import save_file

            save_file(
                {name: tensor.contiguous() for name, tensor in snapshot["policy_lora"].items()},
                policy_dir / "adapter_model.safetensors",
            )
            (policy_dir / "adapter_config.json").write_text(
                json.dumps(snapshot.get("adapter_config", {}), indent=2), encoding="utf-8",
            )
        torch.save(snapshot["optimizer"], temp / "optimizer.pt")
        torch.save(snapshot["lr_scheduler"], temp / "lr_scheduler.pt")
        torch.save(snapshot["ema"], temp / "ema.pt")
        torch.save(snapshot["rng"], temp / "rng_state.pt")
        torch.save(snapshot["sampler"], temp / "sampler_state.pt")
        with (temp / "trainer_state.json").open("w", encoding="utf-8") as handle:
            json.dump(snapshot["trainer_state"], handle, ensure_ascii=False, indent=2)
        with (temp / "train_config.json").open("w", encoding="utf-8") as handle:
            json.dump(snapshot["config"], handle, ensure_ascii=False, indent=2, default=str)
        (temp / "COMPLETE").write_text("ok\n", encoding="utf-8")
        os.replace(temp, final)
    return final


def save_checkpoint(
    output_dir: str | Path,
    *,
    step: int,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    ema_state: dict[str, Any],
    sampler_state: dict[str, Any],
    trainer_state: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """Synchronous save (back-compat wrapper): build a snapshot then write it."""
    snapshot = build_checkpoint_snapshot(
        step=step, policy=policy, optimizer=optimizer, lr_scheduler=lr_scheduler,
        ema_state=ema_state, sampler_state=sampler_state, trainer_state=trainer_state, config=config,
    )
    return save_checkpoint_from_snapshot(output_dir, step=step, snapshot=snapshot)


class AsyncCheckpointWriter:
    """Background checkpoint writer.

    The main thread builds a CPU snapshot (fast memcpy) and submits it; the
    writer thread does the (slow) serialization + disk I/O with an atomic
    temp-dir rename, so rank 0 is never blocked on checkpoint I/O and a crash
    can never leave a half-written checkpoint as the final path. Only the most
    recent *pending* checkpoint is kept (an older queued-but-unsaved one is
    dropped); `close()` flushes whatever is queued.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pending: tuple[Path, int, dict[str, Any]] | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="awm-checkpoint-writer", daemon=True)
        self._thread.start()

    def submit(self, output_dir: str | Path, step: int, snapshot: dict[str, Any]) -> None:
        with self._cond:
            if self._closed:
                raise RuntimeError("async checkpoint writer is closed")
            self._pending = (Path(output_dir), int(step), snapshot)
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None:
                    if self._closed:
                        return
                    self._cond.wait()
                output_dir, step, snapshot = self._pending
                self._pending = None
            try:
                save_checkpoint_from_snapshot(output_dir, step=step, snapshot=snapshot)
            except Exception as exc:  # noqa: BLE001
                # 训练不受影响，仅打印错误（stdout 进入 train_*.log）。
                print(f"[AsyncCheckpointWriter] checkpoint save failed for step {step}: {exc}", flush=True)

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify()
        self._thread.join()

    def __enter__(self) -> "AsyncCheckpointWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = Path(path)
    if not (checkpoint / "COMPLETE").is_file():
        raise ValueError(f"incomplete checkpoint: {checkpoint}")
    with (checkpoint / "trainer_state.json").open("r", encoding="utf-8") as handle:
        trainer_state = json.load(handle)
    with (checkpoint / "train_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return {
        "path": checkpoint,
        "optimizer": torch.load(checkpoint / "optimizer.pt", map_location=map_location),
        "lr_scheduler": torch.load(checkpoint / "lr_scheduler.pt", map_location=map_location),
        "ema": torch.load(checkpoint / "ema.pt", map_location="cpu"),
        "rng": torch.load(checkpoint / "rng_state.pt", map_location="cpu"),
        "sampler": torch.load(checkpoint / "sampler_state.pt", map_location="cpu"),
        "trainer_state": trainer_state,
        "config": config,
    }
