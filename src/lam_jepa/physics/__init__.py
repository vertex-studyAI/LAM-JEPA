"""Small deterministic primitives for predeclared physics research protocols.

These utilities are intentionally independent from historical ARC experiment code.
"""

from .masking import make_observation_mask
from .simulators import DampedOscillatorParams, simulate_damped_oscillator, split_parameter_grid

__all__ = [
    "DampedOscillatorParams",
    "make_observation_mask",
    "simulate_damped_oscillator",
    "split_parameter_grid",
]
