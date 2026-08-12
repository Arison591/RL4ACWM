from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RewardRecord:
    total_reward: float | None
    action_reward: float | None
    geometry_reward: float | None = None
    action_metrics: dict[str, Any] = field(default_factory=dict)
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    error: str | None = None
    action_camera: str = "head"
    geometry_cameras: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalRecord:
    sample_id: str
    seed: int | None
    reward: RewardRecord
    credit: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reward"] = self.reward.to_dict()
        return data
