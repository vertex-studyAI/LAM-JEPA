from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
import numpy as np


@dataclass
class SignificanceResult:
    mean: float
    ci_low: float
    ci_high: float
    p_value: float
    effect_size: float | None
    mean_a: float | None = None
    mean_b: float | None = None
    n_pairs: int = 0
    estimand: str = "paired_mean_difference"
    effect_size_kind: str = "mean_difference_over_average_within_sample_sd"


def _sample(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values))
    if arr.ndim != 1 or arr.size == 0 or arr.dtype.kind not in "fiu":
        raise ValueError("samples must be nonempty one-dimensional real numeric vectors")
    arr = arr.astype(float)
    if not np.isfinite(arr).all():
        raise ValueError("samples must contain only finite values")
    return arr


def _pairs(a: Sequence[float], b: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = _sample(a), _sample(b)
    if left.shape != right.shape:
        raise ValueError("paired samples must have equal length and be aligned by independent unit")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = left - right
    if not np.isfinite(difference).all():
        raise ValueError("paired differences exceed the finite numerical range")
    return left, right, difference


def _count(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _confidence(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("confidence must be a finite number strictly between zero and one")
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("confidence must be a finite number strictly between zero and one")
    return float(value)


def _finite_mean(values: np.ndarray) -> float:
    """Average finite values without overflowing an intermediate sum."""
    scale = float(np.max(np.abs(values)))
    return float((values / scale).mean() * scale) if scale else 0.0


def _finite_std(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values)))
    if len(values) < 2 or scale == 0.0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        result = float((values / scale).std(ddof=1) * scale)
    if not math.isfinite(result):
        raise ValueError("sample standard deviation exceeds the finite numerical range")
    return result


def bootstrap_ci(values: Sequence[float], num_bootstrap: int = 2000, confidence: float = 0.95, seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean over independent input units."""
    arr = _sample(values)
    num_bootstrap = _count(num_bootstrap, "num_bootstrap")
    confidence = _confidence(confidence)
    # Both sums and quantile interpolation can overflow on finite observations.
    # Resample in bounded units and restore units only after both reductions.
    scale = float(np.max(np.abs(arr)))
    if scale == 0.0:
        return 0.0, 0.0
    arr = arr / scale
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(num_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats.append(float(sample.mean()))
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(stats, alpha) * scale)
    high = float(np.quantile(stats, 1.0 - alpha) * scale)
    return low, high


def paired_permutation_test(a: Sequence[float], b: Sequence[float], num_permutations: int = 10000, seed: int = 7) -> float:
    """Two-sided Monte Carlo sign-flip test with a plus-one correction.

    Pairs must be aligned independent units and sign-exchangeable under the
    null. This is a sampled randomization test, not exact enumeration. Scaling
    by the largest absolute difference makes tail comparisons independent of
    measurement units and avoids an absolute 1e-12 scientific effect floor.
    """
    _, _, diff = _pairs(a, b)
    num_permutations = _count(num_permutations, "num_permutations")
    scale = float(np.max(np.abs(diff)))
    if scale == 0.0:
        return 1.0
    diff = diff / scale
    observed = abs(float(diff.mean()))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(num_permutations):
        signs = rng.choice([-1.0, 1.0], size=diff.size)
        perm = abs((diff * signs).mean())
        # Permit only relative rounding error when the observed statistic is
        # nonzero; a fixed absolute tolerance would alter small-effect tests.
        if perm >= observed or math.isclose(float(perm), observed, rel_tol=8 * np.finfo(float).eps, abs_tol=0.0):
            count += 1
    return float((count + 1) / (num_permutations + 1))


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Difference standardized by average within-sample variance (not d_z).

    Return None when the standardized effect is undefined; a numerical variance
    floor would manufacture a finite effect size and depend on the units.
    """
    a, b, difference = _pairs(a, b)
    if len(a) < 2:
        raise ValueError("effect size requires at least two paired units")
    scale = float(max(np.max(np.abs(a)), np.max(np.abs(b))))
    if scale == 0.0:
        return None
    pooled = math.sqrt(float(((a / scale).var(ddof=1) + (b / scale).var(ddof=1)) / 2.0))
    if pooled == 0.0:
        return None
    return float((difference / scale).mean() / pooled)


def summarize_seeds(seed_results: dict[str, Sequence[float]], confidence: float = 0.95) -> dict[str, dict[str, float]]:
    summary = {}
    for key, values in seed_results.items():
        values = _sample(values)
        mean = _finite_mean(values)
        std = _finite_std(values)
        ci_low, ci_high = bootstrap_ci(values, confidence=confidence)
        summary[key] = {"mean": mean, "std": std, "ci_low": ci_low, "ci_high": ci_high, "n": len(values)}
    return summary


def significance_report(a: Sequence[float], b: Sequence[float], name_a: str = "A", name_b: str = "B") -> SignificanceResult:
    """Report a paired A-minus-B comparison over aligned independent units.

    ``mean`` and its bootstrap interval describe the difference tested by the
    paired p-value. Descriptive means of each method are separate fields. This
    corrects the former mixture of an A-only interval and a comparison p-value.
    ``name_a``/``name_b`` remain accepted for source compatibility.
    """
    a, b, difference = _pairs(a, b)
    if len(a) < 2:
        raise ValueError("a comparison report requires at least two independent pairs")
    mean = _finite_mean(difference)
    ci_low, ci_high = bootstrap_ci(difference)
    p = paired_permutation_test(a, b)
    effect = cohens_d(a, b)
    return SignificanceResult(
        mean=mean, ci_low=ci_low, ci_high=ci_high, p_value=p, effect_size=effect,
        mean_a=_finite_mean(a), mean_b=_finite_mean(b), n_pairs=len(a),
    )
