from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentSummary:
    mean: torch.Tensor
    variance: torch.Tensor
    confidence: torch.Tensor
    metadata: dict[str, Any]


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2, dropout: float = 0.1) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(max(depth - 1, 0)):
        layers.extend([nn.Linear(d, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)])
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def safe_mean(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device if isinstance(x, torch.Tensor) else None)
    return x.mean(dim=dim) if dim is not None else x.mean()


def clip_prob(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return x.clamp(min=eps, max=1.0 - eps)


def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    return x


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    if mask is None:
        return torch.softmax(logits, dim=dim)
    mask = mask.to(dtype=logits.dtype)
    logits = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=dim)


def weighted_average(values: torch.Tensor, weights: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    weights = weights / weights.sum(dim=dim, keepdim=True).clamp_min(eps)
    return (values * weights).sum(dim=dim)
