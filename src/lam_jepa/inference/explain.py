from __future__ import annotations
import torch


@torch.no_grad()
def explain(model, tokens: torch.Tensor, numeric_x: torch.Tensor | None = None, steps: int = 3):
    out = model(tokens, numeric_x=numeric_x, steps=steps)
    return {
        "prediction": out["logits"].argmax(dim=-1),
        "confidence": out["confidence"].squeeze(-1),
        "verifier": out["verifier"].squeeze(-1),
        "rubric": out["rubric"],
        "actions": out["actions"],
        "trajectory": out["traj"],
    }
