from __future__ import annotations
import torch


@torch.no_grad()
def rollout(model, tokens: torch.Tensor, numeric_x: torch.Tensor | None = None, steps: int = 3):
    out = model(tokens, numeric_x=numeric_x, steps=steps)
    return out["traj"], out["actions"], out["logits"]
