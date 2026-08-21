"""Deterministic PHY-JEPA phase-0 dataset, manifest, and linear-probe utilities.

This module is deliberately outcome-agnostic. It materializes the predeclared
64-sample damped-oscillator sanity dataset and a simple linear probe without
changing any historical ARC result or creating a positive-result claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from .simulators import DampedOscillatorParams, simulate_damped_oscillator, split_parameter_grid


@dataclass(frozen=True)
class OscillatorPhase0Config:
    """Frozen generator settings for the 64-sample PHY-JEPA phase-0 gate."""

    damping_ratios: tuple[float, ...] = (0.02, 0.06, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30)
    natural_frequencies: tuple[float, ...] = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2)
    duration: float = 6.0
    time_steps: int = 121
    x0: float = 1.0
    v0: float = 0.0
    split_seed: int = 17
    train_fraction: float = 0.70
    validation_fraction: float = 0.15

    def validate(self) -> None:
        if len(self.damping_ratios) * len(self.natural_frequencies) != 64:
            raise ValueError("phase-0 requires exactly 64 unique parameter combinations")
        if len(set(self.damping_ratios)) != len(self.damping_ratios):
            raise ValueError("damping_ratios must be unique")
        if len(set(self.natural_frequencies)) != len(self.natural_frequencies):
            raise ValueError("natural_frequencies must be unique")
        if any((not np.isfinite(value)) or value < 0.0 or value >= 1.0 for value in self.damping_ratios):
            raise ValueError("all damping ratios must be finite and in [0, 1)")
        if any((not np.isfinite(value)) or value <= 0.0 for value in self.natural_frequencies):
            raise ValueError("all natural frequencies must be finite and positive")
        if not np.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("duration must be finite and positive")
        if self.time_steps < 3:
            raise ValueError("time_steps must be at least 3")
        if not np.isfinite(self.x0) or not np.isfinite(self.v0):
            raise ValueError("initial conditions must be finite")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_sha256(states: np.ndarray) -> str:
    stable = np.asarray(states, dtype="<f8", order="C")
    return hashlib.sha256(stable.tobytes(order="C")).hexdigest()


def generate_oscillator_phase0_dataset(
    config: OscillatorPhase0Config = OscillatorPhase0Config(),
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Generate the exact 64-trajectory phase-0 dataset and provenance manifest.

    Returns
    -------
    features:
        Flattened ``[position, velocity]`` trajectories with shape
        ``(64, time_steps * 2)``. These are a transparent raw-state baseline,
        not JEPA embeddings.
    targets:
        Hidden ``[damping_ratio, natural_frequency]`` values with shape
        ``(64, 2)``.
    manifest:
        Canonical generator settings, split membership, per-trajectory hashes,
        and a digest over all entries. No test labels are used to alter splits.
    """

    config.validate()
    grid = [
        (float(zeta), float(omega))
        for zeta in config.damping_ratios
        for omega in config.natural_frequencies
    ]
    splits = split_parameter_grid(
        grid,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        seed=config.split_seed,
    )
    membership = {params: split_name for split_name, values in splits.items() for params in values}
    if len(membership) != 64:
        raise RuntimeError("phase-0 split lost or duplicated a parameter combination")

    t = np.linspace(0.0, config.duration, config.time_steps, dtype=np.float64)
    feature_rows: list[np.ndarray] = []
    target_rows: list[tuple[float, float]] = []
    entries: list[dict[str, object]] = []

    for index, params in enumerate(grid):
        zeta, omega = params
        states = simulate_damped_oscillator(
            DampedOscillatorParams(damping_ratio=zeta, natural_frequency=omega),
            t,
            x0=config.x0,
            v0=config.v0,
        )
        feature_rows.append(states.reshape(-1))
        target_rows.append(params)
        entries.append(
            {
                "sample_id": f"osc-{index:02d}",
                "damping_ratio": zeta,
                "natural_frequency": omega,
                "split": membership[params],
                "state_shape": list(states.shape),
                "state_sha256": _state_sha256(states),
            }
        )

    features = np.stack(feature_rows, axis=0)
    targets = np.asarray(target_rows, dtype=np.float64)
    manifest_core: dict[str, object] = {
        "protocol_id": "PHY-JEPA-v1",
        "phase": 0,
        "status": "GENERATED_NO_OUTCOME_CLAIM",
        "feature_semantics": "flattened raw position/velocity trajectory; not a JEPA representation",
        "target_semantics": ["damping_ratio", "natural_frequency"],
        "config": asdict(config),
        "time_grid_sha256": _state_sha256(t),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "samples": entries,
    }
    manifest = dict(manifest_core)
    manifest["manifest_sha256"] = _canonical_sha256(manifest_core)
    return features, targets, manifest


def split_indices_from_manifest(manifest: Mapping[str, object], split_name: str) -> np.ndarray:
    """Return sample indices for one declared split without re-randomizing it."""

    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split_name must be train, validation, or test")
    samples = manifest.get("samples")
    if not isinstance(samples, Sequence):
        raise ValueError("manifest samples are missing")
    indices = [
        index
        for index, sample in enumerate(samples)
        if isinstance(sample, Mapping) and sample.get("split") == split_name
    ]
    if not indices:
        raise ValueError(f"manifest contains no {split_name} samples")
    return np.asarray(indices, dtype=np.int64)


@dataclass(frozen=True)
class LinearProbe:
    """Least-squares linear probe fitted only on an explicitly supplied split."""

    coefficients: np.ndarray
    intercept: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.coefficients.shape[0]:
            raise ValueError("features have incompatible shape for this probe")
        predictions = x @ self.coefficients + self.intercept
        if not np.all(np.isfinite(predictions)):
            raise FloatingPointError("linear probe produced non-finite predictions")
        return predictions


def fit_linear_probe(features: np.ndarray, targets: np.ndarray) -> LinearProbe:
    """Fit an unregularized linear probe with an intercept via least squares."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("features and targets must be aligned 2D arrays with at least two samples")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("features and targets must contain only finite values")

    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    solution, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return LinearProbe(coefficients=solution[:-1], intercept=solution[-1])


def median_normalized_parameter_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    parameter_ranges: Sequence[float],
) -> float:
    """Median absolute error normalized by predeclared parameter ranges."""

    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)
    ranges = np.asarray(parameter_ranges, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 2:
        raise ValueError("predictions and targets must be aligned 2D arrays")
    if ranges.shape != (true.shape[1],) or np.any(~np.isfinite(ranges)) or np.any(ranges <= 0.0):
        raise ValueError("parameter_ranges must contain one finite positive value per target")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(true)):
        raise ValueError("predictions and targets must contain only finite values")
    normalized = np.abs(pred - true) / ranges
    return float(np.median(normalized))
