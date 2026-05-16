from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import torch

from ..model import LAMJEPA, LAMJEPAConfig
from ..callbacks.checkpointing.load import load_checkpoint


def load_model(checkpoint: str | Path, device: str = "cpu") -> LAMJEPA:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = LAMJEPAConfig(**ckpt.get("config", {}))
    model = LAMJEPA(cfg).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    return model


@torch.no_grad()
def predict(model: LAMJEPA, tokens: torch.Tensor, numeric_x: torch.Tensor | None = None, steps: int = 0) -> Dict[str, Any]:
    out = model(tokens, numeric_x=numeric_x, steps=steps)
    probs = torch.softmax(out["logits"], dim=-1)
    pred = probs.argmax(dim=-1)
    return {
        "pred": pred,
        "probabilities": probs,
        "confidence": out["confidence"],
        "verifier": out["verifier"],
        "rubric": out["rubric"],
        "trajectory": out["traj"],
    }
