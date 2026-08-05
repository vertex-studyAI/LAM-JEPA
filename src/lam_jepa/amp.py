from __future__ import annotations

from contextlib import AbstractContextManager

import torch


def make_grad_scaler(*, enabled: bool):
    """Create a CUDA gradient scaler without using deprecated AMP entry points.

    PyTorch 2.3+ exposes the device-aware scaler under ``torch.amp``. The
    fallback preserves compatibility with the project's declared older 2.x
    floor and is reached only when the modern API is unavailable.
    """

    grad_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if grad_scaler is not None:
        return grad_scaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(*, device: torch.device, enabled: bool) -> AbstractContextManager:
    """Return the device-aware autocast context recommended by modern PyTorch."""

    return torch.autocast(device_type=device.type, enabled=enabled)
