from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryItem:
    key: torch.Tensor
    value: torch.Tensor
    salience: float
    timestamp: int
    tags: Optional[torch.Tensor] = None


class SparseMemory(nn.Module):
    def __init__(self, dim: int, capacity: int = 2048, top_k: int = 16):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.top_k = top_k
        self.register_parameter("keys", nn.Parameter(torch.randn(capacity, dim)))
        self.register_parameter("values", nn.Parameter(torch.randn(capacity, dim)))
        self.register_buffer("usage", torch.zeros(capacity))
        self.register_buffer("age", torch.zeros(capacity))

        self.query = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 3, dim)
        self.out = nn.Linear(dim, dim)

    def retrieve(self, z: torch.Tensor) -> torch.Tensor:
        q = self.query(z)
        sim = q @ self.keys.t() / math.sqrt(self.dim)
        top = sim.topk(k=min(self.top_k, self.capacity), dim=-1)
        idx = top.indices
        weights = torch.softmax(top.values, dim=-1)
        values = self.values[idx]
        r = (weights.unsqueeze(-1) * values).sum(dim=-2)
        return r

    def gated_correction(self, z: torch.Tensor, r: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        if u is None:
            u = torch.zeros_like(z)
        g = torch.sigmoid(self.gate(torch.cat([z, r, u], dim=-1)))
        return self.out(z + g * r)

    @torch.no_grad()
    def write(self, z: torch.Tensor, value: torch.Tensor, salience: torch.Tensor) -> None:
        if z.ndim == 1:
            z = z.unsqueeze(0)
            value = value.unsqueeze(0)
            salience = salience.unsqueeze(0)
        n = min(z.size(0), self.capacity)
        if n == 0:
            return
        score = salience.flatten()
        idx = torch.topk(score, k=n).indices
        self.keys.data[:n] = z[idx]
        self.values.data[:n] = value[idx]
        self.usage[:n] += 1
