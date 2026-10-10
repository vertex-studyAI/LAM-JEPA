from __future__ import annotations

import torch

from ._evaluation import evaluation_mode, validate_steps


@torch.no_grad()
def rollout(model, tokens: torch.Tensor, numeric_x: torch.Tensor | None = None, steps: int = 3):
    validate_steps(steps)
    with evaluation_mode(model):
        out = model(tokens, numeric_x=numeric_x, steps=steps, sample_rollout=False)
        return out["traj"], out["actions"], out["logits"]
