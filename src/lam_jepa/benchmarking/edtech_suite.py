from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Sequence
import json
import random

import numpy as np
import torch

from ..analysis.statistics import summarize_seed_runs
from ..data import SUPPORTED_TASKS
from ..model import LAMJEPA, LAMJEPAConfig
from ..trainers.trainer import Trainer, TrainerConfig
from ..utils import set_seed
from .evaluation_sampling import TARGET_SEMANTICS, evaluation_sample_digest, sample_evaluation_batch


EDTECH_TASKS = SUPPORTED_TASKS


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


def _forward_without_advancing_sampler_rng(
    model: LAMJEPA,
    tokens: torch.Tensor,
    numeric_x: torch.Tensor | None,
) -> dict:
    """Run inference while preserving RNG streams used by benchmark sampling."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return model(tokens, numeric_x=numeric_x, steps=0)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@torch.no_grad()
def evaluate_model(
    model: LAMJEPA,
    cfg: LAMJEPAConfig,
    tasks: Sequence[str] = EDTECH_TASKS,
    batch_size: int = 64,
    batches: int = 8,
) -> Dict[str, Dict[str, float | int | str]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if batches < 1:
        raise ValueError("batches must be at least 1")

    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model must contain at least one parameter") from exc

    out: Dict[str, Dict[str, float | int | str]] = {}
    for task in tasks:
        accs, confs = [], []
        labels_all = []
        ordered_fingerprints: list[str] = []
        ordered_labels: list[int] = []
        input_fingerprints: set[str] = set()
        prompts: set[str] = set()

        for _ in range(batches):
            batch = sample_evaluation_batch(task, batch_size=batch_size, vocab_size=cfg.vocab_size)
            tokens = batch.tokens.to(device)
            numeric_x = batch.numeric_x.to(device) if batch.numeric_x is not None else None
            labels = batch.labels.to(device)

            res = _forward_without_advancing_sampler_rng(model, tokens, numeric_x)
            pred = res["logits"].argmax(dim=-1)
            correct = (pred == labels).float()
            accs.append(float(correct.mean().item()))
            confs.append(float(res["confidence"].mean().item()))
            labels_cpu = labels.detach().cpu().reshape(-1)
            labels_all.append(labels_cpu)
            ordered_labels.extend(int(value) for value in labels_cpu.tolist())

            fingerprints = batch.metadata.get("input_fingerprints", [])
            if isinstance(fingerprints, list):
                normalized = [str(value) for value in fingerprints]
                ordered_fingerprints.extend(normalized)
                input_fingerprints.update(normalized)
            batch_prompts = batch.metadata.get("prompts", [])
            if isinstance(batch_prompts, list):
                prompts.update(str(value) for value in batch_prompts if value)

        labels = torch.cat(labels_all)
        out[task] = {
            "accuracy": float(np.mean(accs)),
            "confidence": float(np.mean(confs)),
            "n": int(labels.numel()),
            "unique_inputs": len(input_fingerprints),
            "unique_labels": int(torch.unique(labels).numel()),
            "unique_prompts": len(prompts),
            "sample_digest": evaluation_sample_digest(ordered_fingerprints, ordered_labels),
            "target_semantics": TARGET_SEMANTICS[task],
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
            per_task[t].append(float(vals["accuracy"]))
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
