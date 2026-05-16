from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import torch

from ..model import LAMJEPA, LAMJEPAConfig
from ..trainers.trainer import Trainer, TrainerConfig
from ..utils import set_seed
from .edtech_suite import EDTECH_TASKS, evaluate_model, build_variant_config, save_json as _save_json


def train_variant(
    variant: str = "full",
    steps: int = 200,
    batch_size: int = 64,
    seed: int = 7,
    device: str = "cpu",
    base_cfg: LAMJEPAConfig | None = None,
) -> tuple[LAMJEPA, LAMJEPAConfig, Trainer]:
    set_seed(seed)
    base_cfg = base_cfg or LAMJEPAConfig()
    cfg = build_variant_config(base_cfg, variant)
    model = LAMJEPA(cfg)
    tcfg = TrainerConfig(
        steps=steps,
        batch_size=batch_size,
        lr=3e-4,
        task="mixed",
        seed=seed,
        device=device,
        checkpoint_dir="experiments/checkpoints",
        log_dir="experiments/logs",
        eval_every=max(steps // 4, 1),
        save_every=max(steps // 2, 1),
        amp=False,
    )
    trainer = Trainer(model, cfg, tcfg)
    model = trainer.fit()
    return model, cfg, trainer


@torch.no_grad()
def evaluate_tasks(model: LAMJEPA, cfg: LAMJEPAConfig, tasks: Sequence[str] = EDTECH_TASKS, batches: int = 8, batch_size: int = 64) -> Dict[str, Dict[str, float]]:
    return evaluate_model(model, cfg, tasks=tasks, batch_size=batch_size, batches=batches)


def save_json(path: str | Path, payload: dict) -> Path:
    return _save_json(path, payload)
