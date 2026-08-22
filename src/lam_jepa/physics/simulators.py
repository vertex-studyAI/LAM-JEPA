"""Deterministic low-cost simulators for PHY-JEPA phase-0 checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DampedOscillatorParams:
    """Dimensionless underdamped harmonic-oscillator parameters."""

    damping_ratio: float
    natural_frequency: float

    def validate(self) -> None:
        if not 0.0 <= self.damping_ratio < 1.0:
            raise ValueError("phase-0 simulator requires 0 <= damping_ratio < 1")
        if not np.isfinite(self.natural_frequency) or self.natural_frequency <= 0.0:
            raise ValueError("natural_frequency must be finite and positive")


def _validate_time(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be a one-dimensional array with at least two samples")
    if not np.all(np.isfinite(t)):
        raise ValueError("t must contain only finite values")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be strictly increasing")
    return t


def simulate_damped_oscillator(
    params: DampedOscillatorParams,
    t: np.ndarray,
    *,
    x0: float = 1.0,
    v0: float = 0.0,
) -> np.ndarray:
    """Return ``[position, velocity]`` for an underdamped oscillator.

    The implementation uses the closed-form solution, avoiding numerical-solver
    variability in the protocol's cheapest correctness gate.
    """

    params.validate()
    t = _validate_time(t)
    if not np.isfinite(x0) or not np.isfinite(v0):
        raise ValueError("initial conditions must be finite")

    zeta = float(params.damping_ratio)
    omega_n = float(params.natural_frequency)
    omega_d = omega_n * np.sqrt(1.0 - zeta**2)

    a = float(x0)
    b = (float(v0) + zeta * omega_n * a) / omega_d
    decay = np.exp(-zeta * omega_n * t)
    cos_term = np.cos(omega_d * t)
    sin_term = np.sin(omega_d * t)

    core = a * cos_term + b * sin_term
    position = decay * core
    core_derivative = -a * omega_d * sin_term + b * omega_d * cos_term
    velocity = decay * (core_derivative - zeta * omega_n * core)

    states = np.stack([position, velocity], axis=-1)
    if not np.all(np.isfinite(states)):
        raise FloatingPointError("simulator produced non-finite state values")
    return states


def split_parameter_grid(
    parameters: Iterable[tuple[float, float]],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, list[tuple[float, float]]]:
    """Split complete parameter combinations without trajectory-frame leakage.

    Parameters are de-duplicated before a seeded permutation. The returned sets
    therefore contain disjoint physical parameter combinations.
    """

    unique = sorted({(float(a), float(b)) for a, b in parameters})
    if len(unique) < 3:
        raise ValueError("at least three unique parameter combinations are required")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train + validation fractions must leave a non-empty test fraction")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    shuffled = [unique[int(i)] for i in order]

    n = len(shuffled)
    n_train = max(1, int(np.floor(n * train_fraction)))
    n_validation = max(1, int(np.floor(n * validation_fraction)))
    if n_train + n_validation >= n:
        n_validation = max(1, n - n_train - 1)
    if n_train + n_validation >= n:
        n_train = n - n_validation - 1

    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_validation],
        "test": shuffled[n_train + n_validation :],
    }
