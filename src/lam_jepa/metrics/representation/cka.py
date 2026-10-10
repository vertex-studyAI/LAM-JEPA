from __future__ import annotations

import math

import torch


def _centered_unit_features(value: torch.Tensor, name: str) -> torch.Tensor:
    """Remove a common offset and scale before forming covariance products."""
    matrix = value.detach().to(dtype=torch.float64)
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")

    # Subtract a row before the mean so a large common offset does not erase
    # small, representable differences. Opposite float64 extremes can overflow
    # that subtraction; scale first only in that case.
    shifted = matrix - matrix[:1]
    if not torch.isfinite(shifted).all():
        scaled = matrix / matrix.abs().max()
        shifted = scaled - scaled[:1]
    scale = shifted.abs().max()
    if scale == 0:
        raise ValueError(f"CKA is undefined for constant {name} representations")
    centered = shifted / scale
    centered = centered - centered.mean(dim=0, keepdim=True)
    norm = torch.linalg.vector_norm(centered)
    if norm == 0:
        raise ValueError(f"CKA is undefined for constant {name} representations")
    return centered / norm


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """Biased linear centered-kernel alignment for paired observations.

    Inputs have shape ``(observations, features)`` and may have different
    feature counts. The same observation order is required. Nonzero rescaling,
    translation, and orthogonal feature transforms preserve this statistic.

    Computation uses detached float64 tensors on the input device. Constant
    representations have an undefined denominator and raise ``ValueError``;
    zero must not be mistaken for an observed dissimilarity. ``eps`` is retained
    as a nonnegative roundoff tolerance for the final [0, 1] bound, not an
    absolute covariance regularizer. It does not change the estimand.
    """
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    for name, value in (("x", x), ("y", y)):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        if value.ndim != 2 or value.shape[0] < 2 or value.shape[1] < 1:
            raise ValueError(f"{name} must have at least two rows and one feature")
        if value.is_complex() or value.layout != torch.strided:
            raise ValueError(f"{name} must be a dense real tensor")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of paired observations")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")

    a = _centered_unit_features(x, "x")
    b = _centered_unit_features(y, "y")
    n, dx, dy = a.shape[0], a.shape[1], b.shape[1]
    # Use the smaller equivalent formulation, avoiding feature-square storage
    # for wide, small-sample representation matrices.
    if 3 * n * n < dx * dx + dy * dy + dx * dy:
        gram_a, gram_b = a @ a.T, b @ b.T
        numerator = torch.sum(gram_a * gram_b)
        denominator = torch.linalg.vector_norm(gram_a) * torch.linalg.vector_norm(gram_b)
    else:
        numerator = torch.sum((a.T @ b).square())
        denominator = torch.linalg.vector_norm(a.T @ a) * torch.linalg.vector_norm(b.T @ b)
    score = float((numerator / denominator).item())
    if not math.isfinite(score) or score < -eps or score > 1.0 + eps:
        raise ValueError("CKA reduction exceeded its finite [0, 1] numerical contract")
    return min(1.0, max(0.0, score))
