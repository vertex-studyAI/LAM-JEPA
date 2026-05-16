from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from ..data import sample_batch
from .statistical_eval import summarize_seeds


@dataclass
class BenchmarkResult:
    task: str
    accuracy: float
    confidence: float
    ood_gap: float
    calibration_error: float
    metadata: dict = field(default_factory=dict)


@dataclass
class OODBenchmarkSuite:
    in_domain_tasks: tuple[str, ...] = ("parity", "modadd", "algebra", "chain")
    ood_tasks: tuple[str, ...] = ("gsm8k", "science", "reading", "tutoring", "reasoning")
    batch_size: int = 64
    batches: int = 8

    def evaluate(self, model, cfg, device: str = "cpu") -> dict:
        model.eval()
        results: dict[str, BenchmarkResult] = {}
        for task in self.in_domain_tasks + self.ood_tasks:
            accs, confs, gaps = [], [], []
            for _ in range(self.batches):
                batch = sample_batch(task, batch=self.batch_size, vocab_size=cfg.vocab_size)
                out = model(batch.tokens.to(device), numeric_x=batch.numeric_x.to(device), steps=0)
                pred = out["logits"].argmax(dim=-1)
                correct = (pred == batch.labels.to(device)).float()
                accs.append(float(correct.mean().item()))
                confs.append(float(out.get("confidence", torch.zeros_like(correct.unsqueeze(-1))).mean().item()))
                gaps.append(float(torch.abs(out.get("confidence", torch.zeros_like(correct.unsqueeze(-1))).squeeze(-1) - correct).mean().item()))
            results[task] = BenchmarkResult(task=task, accuracy=float(np.mean(accs)), confidence=float(np.mean(confs)), ood_gap=float(np.mean(gaps)), calibration_error=float(np.mean(gaps)), metadata={"n": self.batch_size * self.batches})
        return {k: v.__dict__ for k, v in results.items()}


def evaluate_ood_suite(model, cfg, batch_size: int = 64, batches: int = 8, device: str = "cpu") -> dict:
    suite = OODBenchmarkSuite(batch_size=batch_size, batches=batches)
    return suite.evaluate(model, cfg, device=device)


def benchmark_summary(results: dict) -> dict:
    tasks = {k: v["accuracy"] for k, v in results.items()}
    ood = {k: v["accuracy"] for k, v in results.items() if k in ("gsm8k", "science", "reading", "tutoring", "reasoning")}
    ind = {k: v["accuracy"] for k, v in results.items() if k not in ood}
    return {
        "task_summary": summarize_seeds(tasks),
        "ood_mean": float(np.mean(list(ood.values()))) if ood else 0.0,
        "in_domain_mean": float(np.mean(list(ind.values()))) if ind else 0.0,
    }
