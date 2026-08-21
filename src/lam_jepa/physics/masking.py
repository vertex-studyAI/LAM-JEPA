"""Deterministic observation masks for PO-OP-JEPA correctness gates."""

from __future__ import annotations

from itertools import product

import numpy as np


def _normalize_shape(spatial_shape: tuple[int, ...]) -> tuple[int, ...]:
    shape = tuple(int(v) for v in spatial_shape)
    if not shape or len(shape) > 2 or any(v <= 0 for v in shape):
        raise ValueError("spatial_shape must contain one or two positive dimensions")
    return shape


def _normalize_patch_size(patch_size: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if isinstance(patch_size, int):
        patch = (patch_size,) * ndim
    else:
        patch = tuple(int(v) for v in patch_size)
    if len(patch) != ndim or any(v <= 0 for v in patch):
        raise ValueError("patch_size must match spatial dimensionality and be positive")
    return patch


def _require_non_degenerate_mask(mask: np.ndarray) -> np.ndarray:
    """Fail closed unless positive missingness preserves both context and targets."""

    if not mask.any():
        raise ValueError("mask configuration leaves no observed context")
    if not (~mask).any():
        raise ValueError("mask configuration leaves no masked target region")
    return mask


def make_observation_mask(
    spatial_shape: tuple[int, ...],
    *,
    missing_fraction: float,
    seed: int,
    mode: str = "random_patch",
    patch_size: int | tuple[int, ...] = 4,
) -> np.ndarray:
    """Return a boolean mask where ``True`` means the location is observed.

    ``random_patch`` masks complete deterministic patches until reaching the
    requested missing-cell target as closely as the patch tiling permits.
    ``contiguous_block`` masks one centered-on-a-seeded-anchor rectangular
    block with approximately the requested area.

    For positive missingness, the function fails closed if the requested grid,
    patch size, and missing fraction cannot preserve at least one observed
    location and at least one masked target location. A zero-context sample is
    not a valid partial-observation example for the predeclared protocol.
    """

    shape = _normalize_shape(spatial_shape)
    if not 0.0 <= missing_fraction < 1.0:
        raise ValueError("missing_fraction must be in [0, 1)")
    if missing_fraction == 0.0:
        return np.ones(shape, dtype=bool)
    if int(np.prod(shape)) < 2:
        raise ValueError("positive missingness requires at least two spatial locations")

    rng = np.random.default_rng(seed)
    mask = np.ones(shape, dtype=bool)
    target_missing = max(1, int(round(mask.size * missing_fraction)))

    if mode == "random_patch":
        patch = _normalize_patch_size(patch_size, len(shape))
        starts_per_dim = [range(0, dim, step) for dim, step in zip(shape, patch)]
        patches: list[tuple[slice, ...]] = []
        for starts in product(*starts_per_dim):
            patches.append(
                tuple(slice(start, min(start + width, dim)) for start, width, dim in zip(starts, patch, shape))
            )
        for index in rng.permutation(len(patches)):
            slc = patches[int(index)]
            currently_observed = int(mask[slc].sum())
            if currently_observed == 0:
                continue
            previous_missing = mask.size - int(mask.sum())
            new_missing = previous_missing + currently_observed
            if previous_missing > 0 and abs(previous_missing - target_missing) < abs(new_missing - target_missing):
                break
            mask[slc] = False
            if new_missing >= target_missing:
                break
        return _require_non_degenerate_mask(mask)

    if mode == "contiguous_block":
        if len(shape) == 1:
            block = max(1, min(shape[0] - 1, target_missing))
            start = int(rng.integers(0, shape[0] - block + 1))
            mask[start : start + block] = False
            return _require_non_degenerate_mask(mask)

        rows, cols = shape
        aspect = rows / cols
        block_rows = max(1, int(round(np.sqrt(target_missing * aspect))))
        block_cols = max(1, int(round(target_missing / block_rows)))
        block_rows = min(block_rows, rows)
        block_cols = min(block_cols, cols)
        if block_rows * block_cols >= mask.size:
            if block_cols > 1:
                block_cols -= 1
            else:
                block_rows -= 1
        row0 = int(rng.integers(0, rows - block_rows + 1))
        col0 = int(rng.integers(0, cols - block_cols + 1))
        mask[row0 : row0 + block_rows, col0 : col0 + block_cols] = False
        return _require_non_degenerate_mask(mask)

    raise ValueError("mode must be 'random_patch' or 'contiguous_block'")
