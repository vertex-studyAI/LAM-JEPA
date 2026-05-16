from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import random


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
    rng = ckpt.get("rng", {})
    if rng:
        try:
            random.setstate(rng["python_random"])
            np.random.set_state(rng["numpy_random"])
            torch.random.set_rng_state(rng["torch_random"])
            if torch.cuda.is_available() and rng.get("torch_cuda_random") is not None:
                torch.cuda.set_rng_state_all(rng["torch_cuda_random"])
        except Exception:
            pass
    return ckpt
