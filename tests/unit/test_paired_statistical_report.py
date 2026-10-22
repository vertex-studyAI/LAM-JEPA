"""Hand-computable paired comparisons, invalid units, and scale invariance."""

import numpy as np
import pytest

from lam_jepa.eval.statistical_eval import (
    bootstrap_ci,
    cohens_d,
    paired_permutation_test,
    significance_report,
    summarize_seeds,
)


def test_report_interval_is_for_the_same_paired_estimand_as_its_test():
    report = significance_report([100.0, 101.0, 102.0], [99.0, 100.0, 101.0])
    assert report.mean == pytest.approx(1.0)
    assert (report.ci_low, report.ci_high) == pytest.approx((1.0, 1.0))
    assert report.mean_a == pytest.approx(101.0)
    assert report.mean_b == pytest.approx(100.0)
    assert report.n_pairs == 3
    assert report.estimand == "paired_mean_difference"


def test_common_offset_changes_neither_paired_interval_nor_test():
    a, b = np.array([1.0, 3.0, 2.0, 7.0]), np.array([2.0, 1.0, 4.0, 3.0])
    before = significance_report(a, b)
    after = significance_report(a + 1000.0, b + 1000.0)
    assert (before.mean, before.ci_low, before.ci_high, before.p_value) == pytest.approx(
        (after.mean, after.ci_low, after.ci_high, after.p_value)
    )


@pytest.mark.parametrize("scale", [1e-20, 1e20])
def test_sign_flip_p_value_is_invariant_to_units(scale):
    a = np.array([1.0, 2.0, 4.0, 8.0, 3.0])
    b = np.zeros_like(a)
    assert paired_permutation_test(scale * a, scale * b, 1000, 31) == paired_permutation_test(a, b, 1000, 31)


def test_sign_flip_matches_direct_enumeration_of_the_sampled_signs():
    a, b = np.array([2., 5., 8.]), np.array([1., 3., 4.])
    rng = np.random.default_rng(17)
    signs = rng.choice([-1., 1.], size=(500, 3))
    tail = sum(abs(float(row @ (a - b))) >= abs(float((a - b).sum())) for row in signs)
    assert paired_permutation_test(a, b, 500, 17) == (tail + 1) / 501


@pytest.mark.parametrize("a,b", [([], []), ([1, 2], [1]), ([[1], [2]], [1, 2]), ([1, np.nan], [1, 2]), ([1, 2], [1, np.inf])])
def test_invalid_pairs_are_not_broadcast_dropped_or_reported_as_null(a, b):
    for function in (paired_permutation_test, cohens_d, significance_report):
        with pytest.raises(ValueError):
            function(a, b)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_resampling_counts_are_positive_integers(count):
    with pytest.raises(ValueError):
        bootstrap_ci([1, 2], num_bootstrap=count)
    with pytest.raises(ValueError):
        paired_permutation_test([1, 2], [0, 0], num_permutations=count)


@pytest.mark.parametrize("confidence", [0, 1, -0.1, np.nan, np.inf, True])
def test_confidence_is_a_finite_interior_probability(confidence):
    with pytest.raises(ValueError):
        bootstrap_ci([1, 2], confidence=confidence)


def test_empty_seed_series_is_missing_evidence_not_zero_performance():
    with pytest.raises(ValueError):
        summarize_seeds({"missing": []})


def test_undefined_standardized_effect_is_explicit():
    assert cohens_d([2, 2], [1, 1]) is None
    report = significance_report([2, 2], [1, 1])
    assert report.effect_size is None
    assert report.mean == 1.0


def test_report_requires_more_than_one_independent_pair():
    with pytest.raises(ValueError, match="two"):
        significance_report([2], [1])


def test_finite_large_comparison_does_not_overflow_reductions():
    report = significance_report([1e308, 1e308], [0.0, 0.0])
    assert (report.mean, report.ci_low, report.ci_high, report.mean_a) == (1e308,) * 4
    assert report.mean_b == 0.0
    assert report.effect_size is None


def test_bootstrap_handles_extreme_opposite_values_without_overflow():
    maximum = np.finfo(float).max
    with np.errstate(over="raise", invalid="raise"):
        assert bootstrap_ci([-maximum, maximum], 500) == (-maximum, maximum)


def test_seed_summary_scales_mean_and_standard_deviation():
    result = summarize_seeds({"large": [1e308, 1e308]})["large"]
    assert result == {"mean": 1e308, "std": 0.0, "ci_low": 1e308, "ci_high": 1e308, "n": 2}


def test_unrepresentable_sample_standard_deviation_is_rejected():
    maximum = np.finfo(float).max
    with pytest.raises(ValueError, match="standard deviation"):
        summarize_seeds({"overflow": [-maximum, maximum]})
