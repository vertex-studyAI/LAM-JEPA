from __future__ import annotations

import torch

from ..data import sample_batch
from .ood_benchmark import OODBenchmarkSuite, benchmark_summary, evaluate_ood_suite
from .statistical_eval import SignificanceResult, bootstrap_ci, cohens_d, paired_permutation_test, significance_report, summarize_seeds
from .trajectory_eval import TrajectoryMetrics, evaluate_trajectory


@torch.no_grad()
def evaluate(model, loader=None, device: str = "cpu"):
    model.eval()
    if loader is None:
        loader = [sample_batch("parity", batch=64, vocab_size=getattr(model, "cfg", getattr(model, "vocab_size", 256)).vocab_size if hasattr(model, "cfg") else 256)]
    correct = 0
    total = 0
    confs = []
    preds_all = []
    labels_all = []
    for batch in loader:
        tokens = batch["tokens"].to(device) if isinstance(batch, dict) else batch.tokens.to(device)
        numeric_x = batch["numeric_x"].to(device) if isinstance(batch, dict) else batch.numeric_x.to(device)
        labels = batch["labels"].to(device) if isinstance(batch, dict) else batch.labels.to(device)
        outputs = model(tokens, numeric_x=numeric_x)
        pred = outputs["logits"].argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
        confs.append(outputs.get("confidence", torch.zeros_like(pred.float().unsqueeze(-1))).detach().cpu())
        preds_all.append(pred.detach().cpu())
        labels_all.append(labels.detach().cpu())
    conf = torch.cat(confs).mean().item() if confs else 0.0
    return {"accuracy": correct / max(total, 1), "confidence": conf, "preds": torch.cat(preds_all) if preds_all else torch.tensor([]), "labels": torch.cat(labels_all) if labels_all else torch.tensor([])}


__all__ = [
    "OODBenchmarkSuite",
    "benchmark_summary",
    "evaluate_ood_suite",
    "SignificanceResult",
    "bootstrap_ci",
    "cohens_d",
    "paired_permutation_test",
    "significance_report",
    "summarize_seeds",
    "TrajectoryMetrics",
    "evaluate_trajectory",
    "evaluate",
]
