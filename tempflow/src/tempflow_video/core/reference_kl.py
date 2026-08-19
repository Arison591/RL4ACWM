from __future__ import annotations

import torch


def closed_form_equal_variance_kl(policy_means: torch.Tensor, reference_means: torch.Tensor,
                                  transition_stds: torch.Tensor) -> torch.Tensor:
    if policy_means.shape != reference_means.shape:
        raise ValueError("policy/reference mean shape mismatch")
    stds = transition_stds.reshape(-1)
    if stds.numel() == 1:
        stds = stds.expand(policy_means.shape[0])
    if stds.numel() != policy_means.shape[0] or torch.any(stds <= 0):
        raise ValueError("one positive std per sample is required")
    reduce_dims = tuple(range(1, policy_means.ndim))
    return ((policy_means - reference_means).square().mean(dim=reduce_dims) /
            (2.0 * stds.square())).mean()


class FrozenReference:
    def __init__(self, module: torch.nn.Module):
        self.module = module.eval()
        self.snapshot = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def assert_unchanged(self) -> None:
        current = self.module.state_dict()
        if current.keys() != self.snapshot.keys() or any(not torch.equal(current[k].detach().cpu(), v)
                                                          for k, v in self.snapshot.items()):
            raise RuntimeError("reference policy changed")

