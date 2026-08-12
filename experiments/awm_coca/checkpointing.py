from __future__ import annotations

import json
import os
import random
import tempfile
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
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"checkpoint_{step}"
    if final.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")
    with tempfile.TemporaryDirectory(prefix=f"checkpoint_{step}_", dir=root) as temp_name:
        temp = Path(temp_name)
        policy_dir = temp / "policy_lora"
        if hasattr(policy, "save_pretrained"):
            policy.save_pretrained(policy_dir)
        else:
            torch.save(policy.state_dict(), temp / "policy.pt")
        torch.save(optimizer.state_dict(), temp / "optimizer.pt")
        torch.save(lr_scheduler.state_dict() if lr_scheduler is not None else {}, temp / "lr_scheduler.pt")
        torch.save(ema_state, temp / "ema.pt")
        torch.save(capture_rng_state(), temp / "rng_state.pt")
        torch.save(sampler_state, temp / "sampler_state.pt")
        with (temp / "trainer_state.json").open("w", encoding="utf-8") as handle:
            json.dump(trainer_state, handle, ensure_ascii=False, indent=2)
        with (temp / "train_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2, default=str)
        (temp / "COMPLETE").write_text("ok\n", encoding="utf-8")
        os.replace(temp, final)
    return final


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
