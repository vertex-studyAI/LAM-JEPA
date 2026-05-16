from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import build_mlp, clip_prob


@dataclass
class MemoryEntry:
    key: torch.Tensor
    value: torch.Tensor
    timestamp: int
    salience: float
    kind: Literal["episodic", "semantic"] = "episodic"
    tags: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


class MemoryRouter(nn.Module):
    """Persistent educational memory with retrieval gating and temporal decay."""

    def __init__(self, dim: int = 128, capacity: int = 256, top_k: int = 8, decay: float = 0.97):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.top_k = top_k
        self.decay = decay
        self.query_proj = build_mlp(dim, dim, dim, depth=2)
        self.key_proj = build_mlp(dim, dim, dim, depth=2)
        self.value_proj = build_mlp(dim, dim, dim, depth=2)
        self.gate = build_mlp(dim * 3, dim, dim, depth=2)
        self.out = build_mlp(dim, dim, dim, depth=2)
        self._episodic: list[MemoryEntry] = []
        self._semantic: list[MemoryEntry] = []
        self._clock = 0

    def _store(self, bucket: list[MemoryEntry], entry: MemoryEntry) -> None:
        bucket.append(entry)
        if len(bucket) > self.capacity:
            bucket.pop(0)

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor, salience: float = 1.0, kind: Literal["episodic", "semantic"] = "episodic", tags: Sequence[str] = (), metadata: dict | None = None) -> None:
        key = key.detach().reshape(-1, self.dim).mean(dim=0)
        value = value.detach().reshape(-1, self.dim).mean(dim=0)
        entry = MemoryEntry(key=key.cpu(), value=value.cpu(), timestamp=self._clock, salience=float(salience), kind=kind, tags=tuple(tags), metadata=dict(metadata or {}))
        self._clock += 1
        if kind == "semantic":
            self._store(self._semantic, entry)
        else:
            self._store(self._episodic, entry)

    def _score_entries(self, query: torch.Tensor, entries: Sequence[MemoryEntry]) -> tuple[torch.Tensor, torch.Tensor]:
        if not entries:
            return query.new_zeros((0,)), query.new_zeros((0, self.dim))
        keys = torch.stack([e.key.to(query.device) for e in entries], dim=0)
        values = torch.stack([e.value.to(query.device) for e in entries], dim=0)
        times = torch.tensor([e.timestamp for e in entries], device=query.device, dtype=query.dtype)
        salience = torch.tensor([e.salience for e in entries], device=query.device, dtype=query.dtype)
        scores = (self.query_proj(query) @ self.key_proj(keys).t()) / math.sqrt(self.dim)
        age = (self._clock - times).clamp_min(0)
        scores = scores + torch.log1p(salience) - age * (1.0 - self.decay)
        return scores, values

    def retrieve(self, query: torch.Tensor, kind: Literal["episodic", "semantic", "all"] = "all", top_k: int | None = None) -> torch.Tensor:
        query = query.reshape(-1, self.dim)
        top_k = top_k or self.top_k
        buckets = []
        if kind in ("episodic", "all"):
            buckets.extend(self._episodic)
        if kind in ("semantic", "all"):
            buckets.extend(self._semantic)
        if not buckets:
            return torch.zeros(query.size(0), self.dim, device=query.device)
        scores, values = self._score_entries(query, buckets)
        k = min(top_k, scores.size(-1))
        top = scores.topk(k=k, dim=-1)
        selected = values[top.indices]
        weights = torch.softmax(top.values, dim=-1)
        return (weights.unsqueeze(-1) * selected).sum(dim=1)

    def route(self, query: torch.Tensor, support: torch.Tensor | None = None, kind: Literal["episodic", "semantic", "all"] = "all") -> torch.Tensor:
        query = query.reshape(-1, self.dim)
        retrieved = self.retrieve(query, kind=kind)
        if support is None:
            support = torch.zeros_like(query)
        gate = torch.sigmoid(self.gate(torch.cat([query, retrieved, support], dim=-1)))
        return self.out(query + gate * retrieved)

    def compress(self, max_items: int = 32) -> None:
        if len(self._episodic) <= max_items:
            return
        merged: list[MemoryEntry] = []
        chunk = max(1, len(self._episodic) // max_items)
        for i in range(0, len(self._episodic), chunk):
            block = self._episodic[i:i + chunk]
            key = torch.stack([e.key for e in block]).mean(dim=0)
            value = torch.stack([e.value for e in block]).mean(dim=0)
            salience = float(sum(e.salience for e in block) / len(block))
            merged.append(MemoryEntry(key=key, value=value, timestamp=block[-1].timestamp, salience=salience, kind="semantic", tags=tuple(sorted(set(t for e in block for t in e.tags))), metadata={"compressed_from": len(block)}))
        self._semantic.extend(merged)
        self._episodic = self._episodic[-max_items:]

    def summary(self) -> dict:
        return {
            "episodic_items": len(self._episodic),
            "semantic_items": len(self._semantic),
            "clock": self._clock,
            "mean_salience": float(torch.tensor([e.salience for e in self._episodic + self._semantic]).mean().item()) if (self._episodic or self._semantic) else 0.0,
        }
