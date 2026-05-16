from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PlanResult:
    trajectory: List[torch.Tensor]
    actions: List[torch.Tensor]
    score: torch.Tensor
    final_state: torch.Tensor


def beam_plan(
    latent_action_model,
    value_head,
    verifier_head,
    z: torch.Tensor,
    steps: int = 6,
    beam_width: int = 4,
    temperature: float = 0.7,
    return_all: bool = False,
):
    beams = [(z, [], [], torch.zeros(z.size(0), device=z.device))]
    for _ in range(steps):
        candidates = []
        for state, traj, acts, score in beams:
            nxt, a, logits, probs = latent_action_model.step(state, temp=temperature)
            v = value_head(nxt).squeeze(-1)
            c = verifier_head(nxt).squeeze(-1)
            new_score = score + 0.6 * v + 0.4 * c - 0.01 * torch.norm(nxt, dim=-1)
            candidates.append((nxt, traj + [nxt], acts + [a], new_score))
        candidates = sorted(candidates, key=lambda x: float(x[3].mean().detach().cpu()), reverse=True)
        beams = candidates[:beam_width]
    final_state, traj, acts, score = beams[0]
    if return_all:
        return PlanResult(trajectory=[z] + traj, actions=acts, score=score, final_state=final_state)
    return final_state
