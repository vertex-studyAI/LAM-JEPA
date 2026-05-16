from __future__ import annotations

import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from .model import LAMJEPA, LAMJEPAConfig
from .losses import total_loss
from .data import sample_batch, Batch
from .train import train
from .eval import evaluate
from .utils import set_seed

__all__ = [
    "LAMJEPA",
    "LAMJEPAConfig",
    "total_loss",
    "sample_batch",
    "Batch",
    "train",
    "evaluate",
    "set_seed",
]
