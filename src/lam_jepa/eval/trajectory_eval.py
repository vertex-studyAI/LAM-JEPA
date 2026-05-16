from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class TrajectoryMetrics:
    smoothness: float
    coherence: float
    entropy: float
    intervention_consistency: float
    curriculum_efficiency: float


def trajectory_smoothness(traj: Sequence[torch.Tensor]) -> float:
    if len(traj) < 2:
        return 0.0
    diffs = [F.mse_loss(a, b).item() for a, b in zip(traj[:-1], traj[1:])]
    return float(sum(diffs) / len(diffs))


def latent_entropy(traj: Sequence[torch.Tensor]) -> float:
    if not traj:
        return 0.0
    x = torch.stack([t.reshape(-1).float() for t in traj])
    p = torch.softmax(x.abs().mean(dim=-1), dim=0)
    return float(-(p * (p + 1e-8).log()).sum().item())


def trajectory_coherence(traj: Sequence[torch.Tensor]) -> float:
    if len(traj) < 2:
        return 1.0
    cos = []
    for a, b in zip(traj[:-1], traj[1:]):
        a = a.reshape(-1)
        b = b.reshape(-1)
        cos.append(F.cosine_similarity(a, b, dim=0).item())
    return float(sum(cos) / len(cos))


def intervention_consistency(predicted: Sequence[float], observed: Sequence[float]) -> float:
    if not predicted or not observed:
        return 0.0
    p = torch.tensor(list(predicted), dtype=torch.float32)
    o = torch.tensor(list(observed), dtype=torch.float32)
    return float(1.0 - torch.abs(p - o).mean().item())


def curriculum_efficiency(scores: Sequence[float], costs: Sequence[float]) -> float:
    if not scores or not costs:
        return 0.0
    s = torch.tensor(list(scores), dtype=torch.float32)
    c = torch.tensor(list(costs), dtype=torch.float32)
    return float((s.mean() / c.mean().clamp_min(1e-6)).item())


def evaluate_trajectory(traj: Sequence[torch.Tensor], predicted: Sequence[float] | None = None, observed: Sequence[float] | None = None, scores: Sequence[float] | None = None, costs: Sequence[float] | None = None) -> TrajectoryMetrics:
    return TrajectoryMetrics(
        smoothness=trajectory_smoothness(traj),
        coherence=trajectory_coherence(traj),
        entropy=latent_entropy(traj),
        intervention_consistency=intervention_consistency(predicted or [], observed or []),
        curriculum_efficiency=curriculum_efficiency(scores or [], costs or []),
    )
