from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _policy_state(policy: torch.nn.Module) -> tuple[str, dict[str, torch.Tensor]]:
    if hasattr(policy, "peft_config"):
        from peft import get_peft_model_state_dict

        return "peft", {
            name: tensor.detach().cpu().clone()
            for name, tensor in get_peft_model_state_dict(policy).items()
        }
    return "full", {
        name: tensor.detach().cpu().clone() for name, tensor in policy.state_dict().items()
    }


def save_tempflow_checkpoint(
    output_root: str | Path,
    *,
    step: int,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    trainer_state: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"checkpoint_{int(step)}"
    if final.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")
    state_kind, state = _policy_state(policy)
    with tempfile.TemporaryDirectory(prefix=f"tempflow_{step}_", dir=root) as temporary:
        target = Path(temporary)
        torch.save({"kind": state_kind, "state": state}, target / "policy.pt")
        torch.save(optimizer.state_dict(), target / "optimizer.pt")
        torch.save(lr_scheduler.state_dict(), target / "lr_scheduler.pt")
        torch.save(_rng_state(), target / "rng_state.pt")
        (target / "trainer_state.json").write_text(
            json.dumps(trainer_state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (target / "train_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (target / "COMPLETE").write_text("ok\n", encoding="utf-8")
        os.replace(target, final)
    return final


def load_tempflow_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict:
    root = Path(path)
    if not (root / "COMPLETE").is_file():
        raise ValueError(f"incomplete checkpoint: {root}")
    return {
        "path": root,
        "policy": torch.load(root / "policy.pt", map_location=map_location, weights_only=False),
        "optimizer": torch.load(root / "optimizer.pt", map_location=map_location, weights_only=False),
        "lr_scheduler": torch.load(
            root / "lr_scheduler.pt", map_location=map_location, weights_only=False
        ),
        "rng": torch.load(root / "rng_state.pt", map_location="cpu", weights_only=False),
        "trainer_state": json.loads((root / "trainer_state.json").read_text(encoding="utf-8")),
        "config": json.loads((root / "train_config.json").read_text(encoding="utf-8")),
    }


def restore_tempflow_checkpoint(
    payload: dict,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
) -> dict:
    policy_payload = payload["policy"]
    if policy_payload["kind"] == "peft":
        from peft import set_peft_model_state_dict

        result = set_peft_model_state_dict(policy, policy_payload["state"])
        if getattr(result, "unexpected_keys", None):
            raise ValueError(f"unexpected PEFT checkpoint keys: {result.unexpected_keys}")
    elif policy_payload["kind"] == "full":
        policy.load_state_dict(policy_payload["state"], strict=True)
    else:
        raise ValueError(f"unknown policy checkpoint kind: {policy_payload['kind']}")
    optimizer.load_state_dict(payload["optimizer"])
    lr_scheduler.load_state_dict(payload["lr_scheduler"])
    rng = payload.get("rng")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng.get("torch_cuda"):
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return payload["trainer_state"]
