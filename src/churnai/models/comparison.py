"""Statistics for comparing two models, offline (paired) and online (split traffic)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

TWO_SIDES = 2
MIN_CLASSES = 2
DIFF_GRID_STEP = 0.001


@dataclass(frozen=True)
class TTestResult:
    mean_diff: float
    t_stat: float
    p_value: float


@dataclass(frozen=True)
class IntervalResult:
    mean_diff: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class McNemarResult:
    a_only_correct: int
    b_only_correct: int
    p_value: float


def corrected_resampled_ttest(diffs: np.ndarray, n_folds: int) -> TTestResult:
    """Nadeau-Bengio t-test on paired fold scores from repeated k-fold CV.

    CV folds share training rows, so fold scores are correlated and a plain t-test
    overstates significance. The variance term adds the test/train size ratio,
    which is 1 / (k - 1) in k-fold CV.
    """
    diffs = np.asarray(diffs, dtype=float)
    mean = float(diffs.mean())
    variance = float(diffs.var(ddof=1))
    if variance == 0:
        # No fold disagrees: either the models are identical or the difference is constant.
        return TTestResult(mean, 0.0, 1.0) if mean == 0 else TTestResult(mean, math.inf, 0.0)
    correction = 1 / len(diffs) + 1 / (n_folds - 1)
    t_stat = mean / math.sqrt(correction * variance)
    return TTestResult(mean, t_stat, float(TWO_SIDES * stats.t.sf(abs(t_stat), len(diffs) - 1)))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment, for when one champion is tested against several models."""
    order = np.argsort(p_values)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def _percentile_interval(samples: np.ndarray, mean_diff: float, confidence: float) -> IntervalResult:
    tail = (1 - confidence) / TWO_SIDES * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return IntervalResult(mean_diff, float(low), float(high))


def bootstrap_auc_diff(
    y: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, iterations: int, seed: int,
    confidence: float = 0.95,
) -> IntervalResult:
    """Paired bootstrap CI of AUC(B) - AUC(A), resampling customers scored by both models."""
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < iterations:
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == MIN_CLASSES:
            samples.append(roc_auc_score(y[idx], proba_b[idx]) - roc_auc_score(y[idx], proba_a[idx]))
    observed = float(roc_auc_score(y, proba_b) - roc_auc_score(y, proba_a))
    return _percentile_interval(np.array(samples), observed, confidence)


def mcnemar_exact(y: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray) -> McNemarResult:
    """Exact McNemar test: only customers on whom exactly one model is right carry information."""
    correct_a, correct_b = labels_a == y, labels_b == y
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    if a_only + b_only == 0:
        return McNemarResult(0, 0, 1.0)
    return McNemarResult(a_only, b_only, float(stats.binomtest(a_only, a_only + b_only, 0.5).pvalue))


def two_proportion_ztest(hits_a: int, n_a: int, hits_b: int, n_b: int) -> tuple[float, float]:
    """Pooled two-sided z-test for independent proportions. Returns (z of B - A, p-value)."""
    pooled = (hits_a + hits_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (hits_b / n_b - hits_a / n_a) / se
    return z, float(TWO_SIDES * stats.norm.sf(abs(z)))


def required_n_per_arm(baseline: float, diff: float, alpha: float, power: float) -> int:
    """Observations per arm for a two-sided two-proportion test to detect ``baseline + diff``."""
    other = baseline + diff
    average = (baseline + other) / TWO_SIDES
    z_alpha = stats.norm.ppf(1 - alpha / TWO_SIDES)
    z_power = stats.norm.ppf(power)
    root = z_alpha * math.sqrt(TWO_SIDES * average * (1 - average)) + z_power * math.sqrt(
        baseline * (1 - baseline) + other * (1 - other)
    )
    return math.ceil(root**2 / diff**2)


def minimum_detectable_diff(baseline: float, n_per_arm: int, alpha: float, power: float) -> float | None:
    """Smallest improvement ``n_per_arm`` observations can detect, or None if none is possible."""
    for diff in np.arange(DIFF_GRID_STEP, 1 - baseline, DIFF_GRID_STEP):
        if required_n_per_arm(baseline, float(diff), alpha, power) <= n_per_arm:
            return round(float(diff), 3)
    return None


def bootstrap_mean_diff(
    a: np.ndarray, b: np.ndarray, iterations: int, seed: int, confidence: float = 0.95
) -> IntervalResult:
    """Unpaired bootstrap CI of mean(B) - mean(A), for arms holding different customers."""
    rng = np.random.default_rng(seed)
    samples = rng.choice(b, (iterations, len(b))).mean(axis=1) - rng.choice(
        a, (iterations, len(a))
    ).mean(axis=1)
    return _percentile_interval(samples, float(b.mean() - a.mean()), confidence)
