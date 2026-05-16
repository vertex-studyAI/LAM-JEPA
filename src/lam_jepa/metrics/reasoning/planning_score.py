from __future__ import annotations
import torch


def planning_score(traj, labels: torch.Tensor) -> float:
    if not traj:
        return 0.0
    final = traj[-1]
    spread = final.detach().float().std(dim=0).mean().item()
    return float(spread)
