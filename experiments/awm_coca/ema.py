from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch


class ParameterEMA:
    def __init__(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]], *, decay: float = 0.99) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must satisfy 0 <= decay < 1")
        self.decay = decay
        self.shadow = OrderedDict(
            (name, parameter.detach().float().cpu().clone()) for name, parameter in named_parameters if parameter.requires_grad
        )
        if not self.shadow:
            raise ValueError("EMA requires trainable parameters")
        self.updates = 0

    @torch.no_grad()
    def update(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> None:
        current = {name: parameter for name, parameter in named_parameters if name in self.shadow}
        if current.keys() != self.shadow.keys():
            raise ValueError("EMA parameter names changed")
        for name, shadow in self.shadow.items():
            shadow.lerp_(current[name].detach().float().cpu(), 1.0 - self.decay)
        self.updates += 1

    def copy_to(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
        backup = {}
        with torch.no_grad():
            for name, parameter in named_parameters:
                if name in self.shadow:
                    backup[name] = parameter.detach().cpu().clone()
                    parameter.copy_(self.shadow[name].to(parameter.device, parameter.dtype))
        return backup

    @staticmethod
    def restore(named_parameters: Iterable[tuple[str, torch.nn.Parameter]], backup: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, parameter in named_parameters:
                if name in backup:
                    parameter.copy_(backup[name].to(parameter.device, parameter.dtype))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        if self.shadow.keys() != state["shadow"].keys():
            raise ValueError("EMA checkpoint parameter names mismatch")
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])
        self.shadow = OrderedDict((name, tensor.detach().float().cpu().clone()) for name, tensor in state["shadow"].items())
