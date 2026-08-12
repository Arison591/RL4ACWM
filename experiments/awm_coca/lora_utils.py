from __future__ import annotations

from typing import Iterable, Sequence

import torch


DEFAULT_ATTENTION_SUFFIXES = ("to_q", "to_k", "to_v", "to_out.0")


def discover_target_modules(model: torch.nn.Module, suffixes: Sequence[str] = DEFAULT_ATTENTION_SUFFIXES) -> list[str]:
    targets = sorted({name for name, module in model.named_modules()
                      if isinstance(module, torch.nn.Linear) and any(name.endswith(suffix) for suffix in suffixes)})
    if not targets:
        raise ValueError(f"no LoRA target modules found for suffixes {tuple(suffixes)}")
    return targets


def install_lora(
    model: torch.nn.Module,
    *,
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.0,
    init: str = "gaussian",
    target_modules: Iterable[str] | None = None,
) -> tuple[torch.nn.Module, list[str]]:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("PEFT is required for AWM-CoCA LoRA training") from exc
    targets = list(target_modules or discover_target_modules(model))
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                        init_lora_weights=init, target_modules=targets)
    wrapped = get_peft_model(model, config)
    return wrapped, targets


def trainable_named_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    result = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not result:
        raise ValueError("model has no trainable parameters")
    return result
