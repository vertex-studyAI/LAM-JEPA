import numpy as np
import pytest

from lam_jepa.physics import (
    OscillatorPhase0Config,
    fit_linear_probe,
    generate_oscillator_phase0_dataset,
    median_normalized_parameter_error,
    split_indices_from_manifest,
)


def test_phase0_generator_is_exactly_64_samples_and_deterministic():
    features_a, targets_a, manifest_a = generate_oscillator_phase0_dataset()
    features_b, targets_b, manifest_b = generate_oscillator_phase0_dataset()

    assert features_a.shape == (64, 242)
    assert targets_a.shape == (64, 2)
    assert np.array_equal(features_a, features_b)
    assert np.array_equal(targets_a, targets_b)
    assert manifest_a == manifest_b
    assert manifest_a["split_counts"] == {"train": 44, "validation": 9, "test": 11}
    assert len({sample["state_sha256"] for sample in manifest_a["samples"]}) == 64


def test_phase0_manifest_splits_are_disjoint_and_cover_every_sample():
    _, _, manifest = generate_oscillator_phase0_dataset()
    train = set(split_indices_from_manifest(manifest, "train").tolist())
    validation = set(split_indices_from_manifest(manifest, "validation").tolist())
    test = set(split_indices_from_manifest(manifest, "test").tolist())

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == set(range(64))


def test_linear_probe_recovers_exact_synthetic_linear_targets():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(32, 5))
    weights = np.array(
        [[1.0, -0.5], [0.2, 0.3], [-1.2, 0.7], [0.0, 2.0], [0.5, 0.1]]
    )
    intercept = np.array([0.25, -1.5])
    targets = features @ weights + intercept

    probe = fit_linear_probe(features, targets)
    predictions = probe.predict(features)
    assert predictions == pytest.approx(targets, abs=1e-10)
    assert (
        median_normalized_parameter_error(
            predictions, targets, parameter_ranges=(1.0, 2.0)
        )
        < 1e-10
    )


def test_phase0_invalid_non_64_grid_fails_closed():
    config = OscillatorPhase0Config(
        damping_ratios=(0.1, 0.2), natural_frequencies=(1.0, 2.0)
    )
    with pytest.raises(ValueError, match="exactly 64"):
        generate_oscillator_phase0_dataset(config)


def test_manifest_rejects_unknown_split():
    _, _, manifest = generate_oscillator_phase0_dataset()
    with pytest.raises(ValueError):
        split_indices_from_manifest(manifest, "holdout")
