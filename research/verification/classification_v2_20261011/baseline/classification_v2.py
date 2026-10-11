"""Prospective streaming multiclass evaluation; frozen ARC analyses are untouched.

All reductions are sample-weighted. Brier is the unscaled multiclass score in
[0, 2]. ECE is descriptive top-label ECE over fixed equal-width bins. Neither
metric establishes calibration by itself or proves experimental independence.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch

_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _labels(value: torch.Tensor, n: int, classes: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError("labels must be a dense integer tensor")
    if value.dtype not in _INTEGER_DTYPES or value.shape != (n,):
        raise ValueError("labels must have exact shape (batch,) and integer dtype; broadcasting is forbidden")
    labels = value.detach().to(device="cpu", dtype=torch.int64)
    if (labels < 0).any() or (classes is not None and (labels >= classes).any()):
        raise ValueError("label is outside the declared class range")
    return labels


class ClassificationAccumulator:
    """Fixed-memory aggregate, independent of the number of observed batches.

    Successful updates retain only scalar sums/means and O(n_bins) bin records;
    logits/predictions/labels and autograd graphs are not retained. Invalid
    updates leave the prior aggregate untouched. Counts and input elements are
    bounded; these bounds do not measure total process or model memory.
    """
    def __init__(self, n_bins: int = 15, *, max_examples: int = 10_000_000,
                 max_batch_elements: int = 10_000_000):
        self.n_bins = _positive_int(n_bins, "n_bins")
        if n_bins > 10_000:
            raise ValueError("n_bins exceeds the 10000-bin limit")
        self.max_examples = _positive_int(max_examples, "max_examples")
        self.max_batch_elements = _positive_int(max_batch_elements, "max_batch_elements")
        self.n = 0
        self.classes: int | None = None
        self.correct = 0
        self.mean_nll = 0.0
        self.mean_brier = 0.0
        self.mean_confidence = 0.0
        self.counts = [0] * n_bins
        self.correct_by_bin = [0] * n_bins
        self.confidence_by_bin = [0.0] * n_bins

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if (not isinstance(logits, torch.Tensor) or logits.layout != torch.strided
                or logits.ndim != 2 or not logits.is_floating_point()):
            raise ValueError("logits must be a dense real floating tensor of shape (batch, classes)")
        batch, classes = logits.shape
        if batch < 1 or classes < 2:
            raise ValueError("each batch needs at least one example and two classes")
        if logits.numel() > self.max_batch_elements or self.n + batch > self.max_examples:
            raise ValueError("classification resource budget exceeded")
        if self.classes is not None and classes != self.classes:
            raise ValueError("the class count changed between batches")
        y = _labels(labels, batch, classes)
        x = logits.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(x).all():
            raise ValueError("logits must be finite")
        log_p = torch.log_softmax(x, dim=1)
        if not torch.isfinite(log_p).all():
            raise ValueError("log probabilities are not representable; batch rejected")
        p = log_p.exp()
        conf, pred = p.max(dim=1)
        hit = pred.eq(y)
        row = torch.arange(batch)
        nll = -log_p[row, y]
        error = p.clone()
        error[row, y] -= 1.0
        brier = error.square().sum(dim=1)
        # Divide before summing nonnegative values: a representable mean need
        # not have a representable raw sum. All metrics below are nonnegative.
        batch_means = [float((values / batch).sum()) for values in (nll, brier, conf)]
        if any(not math.isfinite(v) or v < 0 for v in batch_means):
            raise ValueError("nonfinite aggregate; batch rejected")
        # [j/B,(j+1)/B), final bin includes 1. Argmax ties choose first class.
        indexes = torch.floor(conf * self.n_bins).to(torch.int64).clamp_max(self.n_bins - 1)
        counts = torch.bincount(indexes, minlength=self.n_bins).tolist()
        hits = torch.bincount(indexes[hit], minlength=self.n_bins).tolist()
        sums = torch.bincount(indexes, weights=conf, minlength=self.n_bins).tolist()
        total = self.n + batch
        weights = (self.n / total, batch / total)
        old_means = (self.mean_nll, self.mean_brier, self.mean_confidence)
        means = [old * weights[0] + new * weights[1] for old, new in zip(old_means, batch_means)]
        if any(not math.isfinite(v) for v in means):
            raise ValueError("aggregate mean is not representable")
        # Publish only after all validation/reductions have succeeded.
        self.counts = [a + b for a, b in zip(self.counts, counts)]
        self.correct_by_bin = [a + b for a, b in zip(self.correct_by_bin, hits)]
        self.confidence_by_bin = [a + b for a, b in zip(self.confidence_by_bin, sums)]
        self.mean_nll, self.mean_brier, self.mean_confidence = means
        self.correct += int(hit.sum())
        self.classes = classes
        self.n = total

    def result(self) -> dict[str, Any]:
        if not self.n:
            raise ValueError("no evaluated examples; empty evaluation is not a zero score")
        bins = []
        for index, (count, correct, confidence) in enumerate(
                zip(self.counts, self.correct_by_bin, self.confidence_by_bin)):
            bins.append({"lower": index / self.n_bins, "upper": (index + 1) / self.n_bins,
                         "count": count, "accuracy": correct / count if count else None,
                         "confidence": confidence / count if count else None})
        ece = math.fsum(abs(correct - confidence) for correct, confidence
                        in zip(self.correct_by_bin, self.confidence_by_bin)) / self.n
        return {"schema": "lam.classification-summary.v2", "examples": self.n,
                "classes": self.classes, "accuracy": self.correct / self.n,
                "mean_top1_probability": self.mean_confidence,
                "negative_log_likelihood": self.mean_nll, "brier_multiclass": self.mean_brier,
                "ece_top_label": ece, "ece_bins": bins,
                "aggregation": "sample_weighted", "confidence_source": "softmax(logits)",
                "scientific_clearance": False}


def evaluate_classification_v2(model: torch.nn.Module, loader: Iterable[Any], *,
                               device: str | torch.device = "cpu", n_bins: int = 15,
                               max_examples: int = 10_000_000,
                               max_batch_elements: int = 10_000_000) -> dict[str, Any]:
    """Evaluate the current LAM forward contract without permanent mode changes.

    ``loader`` must be explicit; a synthetic fallback is never substituted.
    Batches may be mappings or objects with tokens, labels and numeric_x.
    LAM's actual forward keywords sample_rollout=False/noise_std=0.0 are used.
    Model weights/device are not moved or restored. A custom model or loader
    that mutates buffers/RNG is not rolled back; this is mode restoration, not
    an arbitrary state transaction. Returned summaries contain no row arrays.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if loader is None:
        raise ValueError("an explicit evaluation loader is required")
    accumulator = ClassificationAccumulator(n_bins, max_examples=max_examples,
                                           max_batch_elements=max_batch_elements)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, Mapping):
                    tokens, labels, numeric = batch.get("tokens"), batch.get("labels"), batch.get("numeric_x")
                else:
                    tokens = getattr(batch, "tokens", None)
                    labels = getattr(batch, "labels", None)
                    numeric = getattr(batch, "numeric_x", None)
                if (not isinstance(tokens, torch.Tensor) or tokens.layout != torch.strided
                        or tokens.ndim != 2 or tokens.dtype not in _INTEGER_DTYPES
                        or min(tokens.shape) < 1):
                    raise ValueError("tokens must be a nonempty dense integer (batch, sequence) tensor")
                if tokens.numel() > max_batch_elements or accumulator.n + tokens.shape[0] > max_examples:
                    raise ValueError("classification resource budget exceeded before model execution")
                _labels(labels, tokens.shape[0], accumulator.classes)
                if numeric is not None:
                    if (not isinstance(numeric, torch.Tensor) or numeric.layout != torch.strided
                            or numeric.ndim not in (1, 2) or numeric.shape[0] != tokens.shape[0]
                            or not numeric.is_floating_point() or not torch.isfinite(numeric).all()
                            or numeric.numel() > max_batch_elements or numeric.numel() == 0):
                        raise ValueError("numeric_x must be finite real batch-aligned data within the budget")
                    numeric = numeric.to(device)
                outputs = model(tokens.to(device), numeric_x=numeric,
                                sample_rollout=False, noise_std=0.0)
                if not isinstance(outputs, Mapping) or "logits" not in outputs:
                    raise ValueError("model must return a mapping containing logits")
                accumulator.update(outputs["logits"], labels)
        return accumulator.result()
    finally:
        # Direct assignment restores intentionally mixed modes without recursive
        # train() calls overwriting descendants that have already been restored.
        for module, training in modes:
            module.training = training
