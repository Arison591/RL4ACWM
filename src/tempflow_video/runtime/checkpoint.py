from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def save_checkpoint(path: str | Path, *, policy: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    trainer_state: dict[str, Any]) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
                "trainer": trainer_state, "rng": rng_state()}, target / "state.pt")
    (target / "COMPLETE").write_text("ok\n", encoding="utf-8")
    return target


def load_checkpoint(path: str | Path, *, policy: torch.nn.Module,
                    optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    root = Path(path)
    if not (root / "COMPLETE").exists():
        raise ValueError("incomplete checkpoint")
    payload = torch.load(root / "state.pt", map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["policy"])
    optimizer.load_state_dict(payload["optimizer"])
    state = payload["rng"]
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    return payload["trainer"]


def save_adapter_checkpoint(path: str | Path, *, policy: torch.nn.Module,
                            optimizer: torch.optim.Optimizer, trainer_state: dict[str, Any]) -> Path:
    """Save PEFT adapter state without materializing the frozen 2B-parameter base model."""
    from peft import get_peft_model_state_dict
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(policy).items()}
    torch.save({"adapter": state, "optimizer": optimizer.state_dict(), "trainer": trainer_state,
                "rng": rng_state()}, target / "state.pt")
    (target / "COMPLETE").write_text("ok\n", encoding="utf-8")
    return target


def load_adapter_checkpoint(path: str | Path, *, policy: torch.nn.Module,
                            optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    from peft import set_peft_model_state_dict
    root = Path(path)
    if not (root / "COMPLETE").is_file():
        raise ValueError("incomplete adapter checkpoint")
    payload = torch.load(root / "state.pt", map_location="cpu", weights_only=False)
    result = set_peft_model_state_dict(policy, payload["adapter"])
    if getattr(result, "unexpected_keys", None):
        raise ValueError(f"unexpected adapter keys: {result.unexpected_keys}")
    optimizer.load_state_dict(payload["optimizer"])
    return payload["trainer"]

