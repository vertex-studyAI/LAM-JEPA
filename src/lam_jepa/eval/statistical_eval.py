from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math
import numpy as np
import torch


@dataclass
class SignificanceResult:
    mean: float
    ci_low: float
    ci_high: float
    p_value: float
    effect_size: float


def bootstrap_ci(values: Sequence[float], num_bootstrap: int = 2000, confidence: float = 0.95, seed: int = 7) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(num_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats.append(float(sample.mean()))
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(stats, alpha))
    high = float(np.quantile(stats, 1.0 - alpha))
    return low, high


def paired_permutation_test(a: Sequence[float], b: Sequence[float], num_permutations: int = 10000, seed: int = 7) -> float:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if a.size != b.size:
        raise ValueError("paired samples must have equal length")
    if a.size == 0:
        return 1.0
    diff = a - b
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(num_permutations):
        signs = rng.choice([-1.0, 1.0], size=diff.size)
        perm = abs((diff * signs).mean())
        if perm >= observed - 1e-12:
            count += 1
    return float((count + 1) / (num_permutations + 1))


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    pooled = math.sqrt(((a.var(ddof=1) + b.var(ddof=1)) / 2.0).clip(min=1e-8))
    return float((a.mean() - b.mean()) / pooled)


def summarize_seeds(seed_results: dict[str, Sequence[float]], confidence: float = 0.95) -> dict[str, dict[str, float]]:
    summary = {}
    for key, values in seed_results.items():
        values = list(map(float, values))
        if not values:
            summary[key] = {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci_low, ci_high = bootstrap_ci(values, confidence=confidence)
        summary[key] = {"mean": mean, "std": std, "ci_low": ci_low, "ci_high": ci_high, "n": len(values)}
    return summary


def significance_report(a: Sequence[float], b: Sequence[float], name_a: str = "A", name_b: str = "B") -> SignificanceResult:
    mean = float(np.mean(list(a))) if len(list(a)) else 0.0
    ci_low, ci_high = bootstrap_ci(a)
    p = paired_permutation_test(a, b)
    effect = cohens_d(a, b)
    return SignificanceResult(mean=mean, ci_low=ci_low, ci_high=ci_high, p_value=p, effect_size=effect)
