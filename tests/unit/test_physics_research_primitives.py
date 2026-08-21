import numpy as np
import pytest

from lam_jepa.physics import (
    DampedOscillatorParams,
    make_observation_mask,
    simulate_damped_oscillator,
    split_parameter_grid,
)


def test_damped_oscillator_respects_initial_conditions_and_shape():
    t = np.linspace(0.0, 4.0, 81)
    states = simulate_damped_oscillator(
        DampedOscillatorParams(damping_ratio=0.15, natural_frequency=2.0),
        t,
        x0=1.25,
        v0=-0.2,
    )

    assert states.shape == (81, 2)
    assert states[0, 0] == pytest.approx(1.25)
    assert states[0, 1] == pytest.approx(-0.2)
    assert np.all(np.isfinite(states))


def test_damping_reduces_late_position_envelope():
    t = np.linspace(0.0, 8.0, 801)
    undamped = simulate_damped_oscillator(
        DampedOscillatorParams(damping_ratio=0.0, natural_frequency=2.0), t
    )
    damped = simulate_damped_oscillator(
        DampedOscillatorParams(damping_ratio=0.25, natural_frequency=2.0), t
    )

    assert np.max(np.abs(damped[-100:, 0])) < np.max(np.abs(undamped[-100:, 0]))


def test_parameter_split_is_seeded_disjoint_and_complete():
    grid = [(zeta, omega) for zeta in (0.05, 0.1, 0.2, 0.3) for omega in (1.0, 1.5, 2.0, 2.5)]
    split_a = split_parameter_grid(grid, seed=17)
    split_b = split_parameter_grid(grid, seed=17)

    assert split_a == split_b
    train = set(split_a["train"])
    validation = set(split_a["validation"])
    test = set(split_a["test"])
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == set(grid)
    assert test


def test_random_patch_mask_is_reproducible_and_nontrivial():
    first = make_observation_mask((16, 16), missing_fraction=0.5, seed=11, patch_size=4)
    second = make_observation_mask((16, 16), missing_fraction=0.5, seed=11, patch_size=4)

    assert np.array_equal(first, second)
    assert first.dtype == np.bool_
    assert first.any()
    assert (~first).any()
    assert (~first).mean() == pytest.approx(0.5, abs=0.07)


def test_contiguous_block_mask_forms_one_missing_interval_in_1d():
    mask = make_observation_mask((40,), missing_fraction=0.25, seed=3, mode="contiguous_block")
    missing = np.flatnonzero(~mask)

    assert len(missing) == 10
    assert np.all(np.diff(missing) == 1)


def test_invalid_phase_zero_inputs_fail_closed():
    t = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError):
        simulate_damped_oscillator(DampedOscillatorParams(damping_ratio=1.0, natural_frequency=2.0), t)
    with pytest.raises(ValueError):
        make_observation_mask((8, 8), missing_fraction=1.0, seed=0)
