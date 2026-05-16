from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import json
import os
import random

import numpy as np
import torch


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.random.get_rng_state(),
        "torch_cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
        "metrics": metrics or {},
        "extra": extra or {},
        "rng": _rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        try:
            payload["scheduler"] = scheduler.state_dict()
        except Exception:
            payload["scheduler"] = None
    torch.save(payload, path)
    return path
