from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from ..model import LAMJEPA, LAMJEPAConfig
from ._evaluation import evaluation_mode, validate_steps


def _checkpoint_config(checkpoint: dict) -> LAMJEPAConfig:
    extra = checkpoint.get("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("checkpoint extra metadata must be an object")
    canonical = extra.get("config")
    legacy = checkpoint.get("config")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise ValueError("checkpoint contains conflicting model configurations")
    config = canonical if canonical is not None else legacy
    if not isinstance(config, dict) or not config:
        raise ValueError("checkpoint does not contain a model configuration")
    try:
        return LAMJEPAConfig(**config)
    except TypeError as exc:
        raise ValueError("checkpoint model configuration is invalid") from exc


def load_model(checkpoint: str | Path, device: str = "cpu") -> LAMJEPA:
    """Load one trusted canonical/legacy checkpoint for inference.

    Training RNG, optimizer and scheduler state are deliberately not restored.
    The model is initialized and populated on CPU before moving to ``device``.
    """
    # Deserialize once: metadata and weights must come from the same snapshot.
    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("model"), dict):
        raise ValueError("checkpoint must contain a model state dictionary")
    cfg = _checkpoint_config(ckpt)
    # Construction consumes random numbers even though its weights are replaced.
    # Preserve the caller's CPU stream rather than replaying training RNG state.
    with torch.random.fork_rng(devices=[]):
        model = LAMJEPA(cfg)
        model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model: LAMJEPA, tokens: torch.Tensor, numeric_x: torch.Tensor | None = None, steps: int = 0) -> Dict[str, Any]:
    validate_steps(steps)
    with evaluation_mode(model):
        out = model(tokens, numeric_x=numeric_x, steps=steps, sample_rollout=False)
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
