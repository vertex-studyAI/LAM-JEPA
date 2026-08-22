"""Small deterministic primitives for predeclared physics research protocols.

These utilities are intentionally independent from historical ARC experiment code.
"""

from .masking import make_observation_mask
from .phase0 import (
    LinearProbe,
    OscillatorPhase0Config,
    fit_linear_probe,
    generate_oscillator_phase0_dataset,
    median_normalized_parameter_error,
    split_indices_from_manifest,
)
from .simulators import DampedOscillatorParams, simulate_damped_oscillator, split_parameter_grid

__all__ = [
    "DampedOscillatorParams",
    "LinearProbe",
    "OscillatorPhase0Config",
    "fit_linear_probe",
    "generate_oscillator_phase0_dataset",
    "make_observation_mask",
    "median_normalized_parameter_error",
    "simulate_damped_oscillator",
    "split_indices_from_manifest",
    "split_parameter_grid",
]
