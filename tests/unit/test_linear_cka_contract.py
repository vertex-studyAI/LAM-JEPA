"""Analytic contracts for the prospective representation-similarity diagnostic."""

import math

import numpy as np
import pytest
import torch

from lam_jepa.metrics.representation.cka import linear_cka


def _reference(x, y):
    """Independent small-matrix definition; only moderate test inputs use it."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    cross = np.linalg.norm(a.T @ b, "fro") ** 2
    return cross / (np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro"))


@pytest.fixture
def features():
    return torch.tensor([[0., 2.], [1., -1.], [3., 4.], [-2., 0.]], dtype=torch.float64)


@pytest.mark.parametrize("scale", [1e-300, 1e-150, 1e-20, 1., 1e20, 1e150, 1e300])
def test_self_similarity_is_invariant_to_finite_nonzero_scale(features, scale):
    assert linear_cka(features * scale, features) == pytest.approx(1., abs=2e-14)


@pytest.mark.parametrize("left_scale,right_scale", [(1e-200, 1e200), (-1e200, 1e-200)])
def test_independent_unit_changes_preserve_nontrivial_similarity(features, left_scale, right_scale):
    y = torch.tensor([[1.], [3.], [-1.], [2.]], dtype=torch.float64)
    expected = _reference(features.numpy(), y.numpy())
    assert 0. < expected < 1.
    assert linear_cka(features * left_scale, y * right_scale) == pytest.approx(expected, abs=2e-14)


def test_float64_structure_survives_large_common_offset(features):
    assert linear_cka(features + 1e12, features) == pytest.approx(1., abs=2e-14)


def test_centering_handles_finite_values_whose_direct_difference_overflows():
    x = torch.tensor([[-1e308], [1e308], [-1e308], [1e308]], dtype=torch.float64)
    y = torch.tensor([[-1.], [1.], [-1.], [1.]], dtype=torch.float64)
    assert linear_cka(x, y) == pytest.approx(1., abs=2e-14)


def test_orthogonal_centered_sample_patterns_have_zero_similarity():
    x = torch.tensor([[-1.], [-1.], [1.], [1.]], dtype=torch.float64)
    y = torch.tensor([[-1.], [1.], [-1.], [1.]], dtype=torch.float64)
    assert linear_cka(x, y) == pytest.approx(0., abs=1e-14)


def test_translation_orthogonal_feature_transform_and_row_pairing(features):
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    transformed = features @ rotation + torch.tensor([19., -7.])
    order = torch.tensor([2, 0, 3, 1])
    assert linear_cka(features[order], transformed[order]) == pytest.approx(1., abs=2e-14)


@pytest.mark.parametrize("n,d,e", [(8, 2, 3), (3, 8, 9)])
def test_feature_and_sample_space_formulations_match_independent_definition(n, d, e):
    x = torch.arange(n * d, dtype=torch.float64).reshape(n, d).sin()
    y = torch.arange(n * e, dtype=torch.float64).reshape(n, e).cos()
    assert linear_cka(x, y) == pytest.approx(_reference(x.numpy(), y.numpy()), abs=2e-14)


@pytest.mark.parametrize("value", [0., 7., 1e300])
def test_constant_representation_is_undefined_not_zero_similarity(features, value):
    with pytest.raises(ValueError, match="constant"):
        linear_cka(torch.full_like(features, value), features)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_representations_are_rejected(features, bad):
    x = features.clone()
    x[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        linear_cka(x, features)


@pytest.mark.parametrize("shape", [(4,), (1, 2), (4, 0), (4, 2, 1), (3, 2)])
def test_invalid_shape_or_pairing_is_rejected(features, shape):
    with pytest.raises(ValueError):
        linear_cka(torch.ones(shape, dtype=torch.float64), features)


def test_complex_input_is_not_silently_projected_to_real(features):
    with pytest.raises(ValueError, match="real"):
        linear_cka(features.to(torch.complex128) + 1j, features)


@pytest.mark.parametrize("eps", [-1., math.nan, math.inf])
def test_invalid_roundoff_tolerance_is_rejected(features, eps):
    with pytest.raises(ValueError, match="eps"):
        linear_cka(features, features, eps=eps)


def test_diagnostic_preserves_inputs_rng_and_autograd_state(features):
    x = features.clone().requires_grad_(True)
    before = x.detach().clone()
    rng = torch.random.get_rng_state().clone()
    assert math.isfinite(linear_cka(x, x))
    assert x.grad is None
    assert torch.equal(x.detach(), before)
    assert torch.equal(torch.random.get_rng_state(), rng)
