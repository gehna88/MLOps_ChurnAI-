"""Statistical tests used by the offline and online A/B comparisons."""

import math

import numpy as np
import pytest
from scipy import stats

from churnai.api.ab import BUCKETS, bucket
from churnai.models.comparison import (
    bootstrap_auc_diff,
    bootstrap_mean_diff,
    corrected_resampled_ttest,
    holm_adjust,
    mcnemar_exact,
    minimum_detectable_diff,
    required_n_per_arm,
    two_proportion_ztest,
)


def test_corrected_ttest_matches_nadeau_bengio_formula() -> None:
    diffs = np.array([0.01, 0.02, 0.0, 0.015, 0.005, 0.01, 0.02, 0.0, 0.01, 0.01])
    folds = 5
    expected_t = diffs.mean() / math.sqrt((1 / len(diffs) + 1 / (folds - 1)) * diffs.var(ddof=1))
    result = corrected_resampled_ttest(diffs, folds)
    assert result.t_stat == pytest.approx(expected_t)
    assert result.p_value == pytest.approx(2 * stats.t.sf(abs(expected_t), len(diffs) - 1))


def test_corrected_ttest_is_more_conservative_than_plain_ttest() -> None:
    diffs = np.random.default_rng(0).normal(0.005, 0.01, 25)
    assert corrected_resampled_ttest(diffs, 5).p_value > stats.ttest_1samp(diffs, 0).pvalue


def test_corrected_ttest_identical_models_show_no_difference() -> None:
    result = corrected_resampled_ttest(np.zeros(25), 5)
    assert (result.mean_diff, result.p_value) == (0.0, 1.0)


def test_holm_adjustment_known_values() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_bootstrap_auc_diff_of_identical_models_is_exactly_zero() -> None:
    rng = np.random.default_rng(0)
    y, proba = rng.integers(0, 2, 200), rng.random(200)
    result = bootstrap_auc_diff(y, proba, proba, 100, seed=0)
    assert (result.mean_diff, result.ci_low, result.ci_high) == (0.0, 0.0, 0.0)


def test_bootstrap_auc_diff_detects_a_clearly_better_model() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    noise, informative = rng.random(500), y + rng.normal(0, 0.5, 500)
    result = bootstrap_auc_diff(y, noise, informative, 200, seed=0)
    assert result.ci_low > 0


def test_mcnemar_counts_only_disagreements() -> None:
    y = np.array([1, 1, 1, 1, 0, 0])
    a = np.array([1, 1, 1, 1, 0, 1])  # wrong on the last row only
    b = np.array([0, 0, 0, 1, 0, 0])  # wrong on the first three rows only
    result = mcnemar_exact(y, a, b)
    assert (result.a_only_correct, result.b_only_correct) == (3, 1)
    assert result.p_value == pytest.approx(stats.binomtest(3, 4, 0.5).pvalue)
    assert mcnemar_exact(y, a, a).p_value == 1.0


def test_two_proportion_ztest_known_value() -> None:
    z, p = two_proportion_ztest(50, 100, 65, 100)
    # Pooled p = 0.575, se = sqrt(0.575 * 0.425 * 0.02) = 0.06991, z = 0.15 / se.
    assert z == pytest.approx(2.1456, abs=1e-3)
    assert p == pytest.approx(0.0319, abs=1e-3)
    assert two_proportion_ztest(0, 10, 0, 10) == (0.0, 1.0)


def test_power_calculation_known_value_and_direction() -> None:
    # Standard textbook case: 50% vs 60% at alpha=0.05, power=0.8 needs about 388 per arm.
    assert required_n_per_arm(0.5, 0.1, 0.05, 0.8) == pytest.approx(388, abs=2)
    assert required_n_per_arm(0.67, 0.05, 0.05, 0.8) > required_n_per_arm(0.67, 0.10, 0.05, 0.8)


def test_minimum_detectable_diff_inverts_the_power_calculation() -> None:
    diff = minimum_detectable_diff(0.67, 200, 0.05, 0.8)
    assert required_n_per_arm(0.67, diff, 0.05, 0.8) <= 200
    assert required_n_per_arm(0.67, diff - 0.002, 0.05, 0.8) > 200
    assert minimum_detectable_diff(0.67, 1, 0.05, 0.8) is None


def test_bootstrap_mean_diff_covers_the_true_difference() -> None:
    rng = np.random.default_rng(0)
    result = bootstrap_mean_diff(rng.normal(10, 1, 500), rng.normal(11, 1, 500), 500, seed=0)
    assert result.ci_low < 1.0 < result.ci_high


def test_bucket_is_stable_and_roughly_uniform() -> None:
    assert bucket("15634602") == bucket("15634602")
    buckets = [bucket(str(i)) for i in range(5000)]
    assert min(buckets) >= 0
    assert max(buckets) < BUCKETS
    assert 0.47 < np.mean(np.array(buckets) < 50) < 0.53
