from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def proposal_metrics(base: torch.Tensor, coca: torch.Tensor, proposal: torch.Tensor) -> dict[str, Any]:
    importance = base / proposal
    entropy = -(proposal * proposal.clamp_min(torch.finfo(proposal.dtype).tiny).log()).sum()
    ess = 1.0 / (proposal * importance.square()).sum()
    return {
        "base": base.detach().cpu().tolist(),
        "q_coca": coca.detach().cpu().tolist(),
        "q": proposal.detach().cpu().tolist(),
        "q_min": float(proposal.min().item()),
        "q_max": float(proposal.max().item()),
        "proposal_entropy": float(entropy.item()),
        "importance_max": float(importance.max().item()),
        "effective_sample_size": float(ess.item()),
    }


class JsonlMetricLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
