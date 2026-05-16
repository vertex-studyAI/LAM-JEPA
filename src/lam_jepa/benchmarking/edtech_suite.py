from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Sequence
import json

import numpy as np
import torch

from ..analysis.statistics import summarize_seed_runs
from ..data import sample_batch
from ..model import LAMJEPA, LAMJEPAConfig
from ..trainers.trainer import Trainer, TrainerConfig
from ..utils import set_seed


EDTECH_TASKS = ("parity", "modadd", "algebra", "chain", "gsm8k", "equation", "science", "reading", "tutoring", "reasoning")


def build_variant_config(base: LAMJEPAConfig, variant: str) -> LAMJEPAConfig:
    cfg = replace(base)
    if variant == "full":
        return cfg
    if variant == "no_memory":
        cfg.use_memory = False
    elif variant == "no_planner":
        cfg.use_planner = False
    elif variant == "no_quant":
        cfg.use_quantizer = False
    elif variant == "no_target":
        cfg.use_target = False
    else:
        raise ValueError(f"unknown variant: {variant}")
    return cfg


def build_config(base: LAMJEPAConfig | None = None) -> LAMJEPAConfig:
    return base or LAMJEPAConfig()


def train_model(
    seed: int = 7,
    steps: int = 120,
    batch_size: int = 64,
    device: str = "cpu",
    task: str = "mixed",
    cfg: LAMJEPAConfig | None = None,
) -> tuple[LAMJEPA, LAMJEPAConfig, Trainer]:
    set_seed(seed)
    cfg = build_config(cfg)
    model = LAMJEPA(cfg)
    trainer = Trainer(
        model,
        cfg,
        TrainerConfig(
            steps=steps,
            batch_size=batch_size,
            lr=3e-4,
            task=task,
            seed=seed,
            device=device,
            checkpoint_dir="experiments/checkpoints",
            log_dir="experiments/logs",
            eval_every=max(steps // 4, 1),
            save_every=max(steps // 2, 1),
            amp=False,
        ),
    )
    model = trainer.fit()
    return model, cfg, trainer


@torch.no_grad()
def evaluate_model(model: LAMJEPA, cfg: LAMJEPAConfig, tasks: Sequence[str] = EDTECH_TASKS, batch_size: int = 64, batches: int = 8) -> Dict[str, Dict[str, float]]:
    model.eval()
    out: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        accs, confs = [], []
        preds_all, labels_all = [], []
        for _ in range(batches):
            batch = sample_batch(task, batch=batch_size, vocab_size=cfg.vocab_size)
            res = model(batch.tokens, numeric_x=batch.numeric_x, steps=0)
            pred = res["logits"].argmax(dim=-1)
            correct = (pred == batch.labels.to(pred.device)).float()
            accs.append(float(correct.mean().item()))
            confs.append(float(res["confidence"].mean().item()))
            preds_all.append(pred.detach().cpu())
            labels_all.append(batch.labels.detach().cpu())
        preds = torch.cat(preds_all)
        labels = torch.cat(labels_all)
        out[task] = {
            "accuracy": float(np.mean(accs)),
            "confidence": float(np.mean(confs)),
            "n": int(labels.numel()),
        }
    return out


def seed_sweep(
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    steps: int = 120,
    batch_size: int = 64,
    device: str = "cpu",
    task: str = "mixed",
) -> dict:
    records = []
    per_task: dict[str, list[float]] = {t: [] for t in EDTECH_TASKS}
    for seed in seeds:
        model, cfg, trainer = train_model(seed=seed, steps=steps, batch_size=batch_size, device=device, task=task)
        scores = evaluate_model(model, cfg, batch_size=batch_size, batches=6)
        records.append({
            "seed": seed,
            "history_tail": trainer.history[-5:],
            "scores": scores,
        })
        for t, vals in scores.items():
            per_task[t].append(vals["accuracy"])
    aggregate = summarize_seed_runs(per_task)
    return {"records": records, "aggregate": aggregate}


def ablation_suite(steps: int = 120, batch_size: int = 64, device: str = "cpu") -> dict:
    base = LAMJEPAConfig()
    variants = ["full", "no_memory", "no_planner", "no_quant", "no_target"]
    results = {}
    for variant in variants:
        cfg = build_variant_config(base, variant)
        model, _, trainer = train_model(seed=7, steps=steps, batch_size=batch_size, device=device, cfg=cfg)
        scores = evaluate_model(model, cfg, batch_size=batch_size, batches=6)
        results[variant] = {
            "scores": scores,
            "history_tail": trainer.history[-5:],
        }
    return results


def save_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
