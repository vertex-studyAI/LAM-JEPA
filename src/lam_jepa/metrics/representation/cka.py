from __future__ import annotations
import torch


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.detach().float() - x.detach().float().mean(dim=0, keepdim=True)
    y = y.detach().float() - y.detach().float().mean(dim=0, keepdim=True)
    hsic = (x.t() @ y).pow(2).sum()
    norm_x = (x.t() @ x).pow(2).sum().clamp_min(eps)
    norm_y = (y.t() @ y).pow(2).sum().clamp_min(eps)
    return float((hsic / torch.sqrt(norm_x * norm_y)).item())
