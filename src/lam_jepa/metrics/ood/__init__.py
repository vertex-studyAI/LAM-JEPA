from __future__ import annotations

import torch


def accuracy_drop(in_domain_acc: float, ood_acc: float) -> float:
    return float(in_domain_acc - ood_acc)


def embedding_shift(z_in: torch.Tensor, z_ood: torch.Tensor) -> float:
    z_in = z_in.detach().float().mean(dim=0)
    z_ood = z_ood.detach().float().mean(dim=0)
    return float(torch.norm(z_in - z_ood).item())
