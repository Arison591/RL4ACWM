from __future__ import annotations

import torch


def group_reward_metrics(action: torch.Tensor, full_db: torch.Tensor, future_db: torch.Tensor,
                         legacy: torch.Tensor) -> dict[str, float]:
    result = {}
    for name, value in (("action_reward", action), ("psnr_full_db", full_db),
                        ("psnr_future_db", future_db), ("legacy_psnr_sigmoid", legacy)):
        value = torch.as_tensor(value, dtype=torch.float64)
        result[f"{name}_mean"] = float(value.mean())
        result[f"{name}_std"] = float(value.std(unbiased=False))
    return result

