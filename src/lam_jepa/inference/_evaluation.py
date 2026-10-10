"""Inference-only mode guards; the frozen model implementation is unchanged."""

from contextlib import contextmanager
from numbers import Integral


def validate_steps(steps: int) -> None:
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
        raise ValueError("steps must be a non-negative integer")


@contextmanager
def evaluation_mode(model):
    """Disable training updates and restore each submodule's previous mode."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        # A top-level train(previous) would erase intentionally mixed modes,
        # such as a training predictor with an evaluation-only target encoder.
        for module, was_training in modes:
            module.training = was_training
