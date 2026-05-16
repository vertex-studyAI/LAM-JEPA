from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

import numpy as np


@dataclass
class EffectSummary:
    mean_a: float
    mean_b: float
    mean_diff: float
    std_diff: float
    cohen_d: float
    ci_low: float
    ci_high: float
    p_value: float


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots.append(float(np.mean(sample)))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    diff = a - b
    sd = np.std(diff, ddof=1) if diff.size > 1 else 0.0
    return float(np.mean(diff) / (sd + 1e-12))


def paired_permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 5000, seed: int = 7) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size == 0:
        return 1.0
    diff = a - b
    observed = abs(float(diff.mean()))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=diff.size)
        perm = abs(float((diff * signs).mean()))
        if perm >= observed - 1e-12:
            count += 1
    return float((count + 1) / (n_perm + 1))


def paired_summary(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> EffectSummary:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    mean_diff = float(diff.mean()) if diff.size else 0.0
    std_diff = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    lo, hi = bootstrap_ci(diff.tolist(), alpha=alpha)
    p = paired_permutation_test(a, b)
    return EffectSummary(
        mean_a=float(a.mean()) if a.size else 0.0,
        mean_b=float(b.mean()) if b.size else 0.0,
        mean_diff=mean_diff,
        std_diff=std_diff,
        cohen_d=cohens_d(a, b),
        ci_low=lo,
        ci_high=hi,
        p_value=p,
    )


def summarize_seed_runs(values: dict[str, Sequence[float]]) -> dict:
    out = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        out[key] = {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "ci95": list(bootstrap_ci(arr.tolist())),
            "n": int(arr.size),
        }
    return out
