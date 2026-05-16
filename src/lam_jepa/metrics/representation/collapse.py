from __future__ import annotations
import torch


def collapse_score(z: torch.Tensor) -> float:
    z = z.detach().float()
    if z.ndim > 2:
        z = z.flatten(1)
    var = z.var(dim=0).mean().item()
    return float(1.0 / (1.0 + var))
