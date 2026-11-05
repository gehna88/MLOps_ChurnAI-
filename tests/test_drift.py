import numpy as np
import pandas as pd
import pytest

from churnai.data.schema import FEATURE_COLUMNS
from churnai.monitoring.drift import DriftMonitor, ReferenceProfile, psi
from churnai.monitoring.metrics import RequestMetrics

BINS = 10
ALERT = 0.2


@pytest.fixture
def features(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    return sample_raw_df[list(FEATURE_COLUMNS)].reset_index(drop=True)


@pytest.fixture
def reference(features: pd.DataFrame) -> ReferenceProfile:
    proba = np.random.default_rng(0).uniform(size=len(features))
    return ReferenceProfile.fit(features, proba, BINS)


def make_monitor(reference: ReferenceProfile, min_samples: int = 50) -> DriftMonitor:
    return DriftMonitor(reference, window_size=500, min_samples=min_samples, alert_threshold=ALERT)


def test_psi_is_zero_for_identical_distributions() -> None:
    p = np.array([0.25, 0.25, 0.5])
    assert psi(p, p) == pytest.approx(0.0)


def test_psi_grows_with_shift() -> None:
    base = np.array([0.5, 0.5])
    assert psi(base, np.array([0.9, 0.1])) > psi(base, np.array([0.6, 0.4])) > 0


def test_discrete_features_profiled_as_categories(reference: ReferenceProfile) -> None:
    assert reference.features["Geography"].is_discrete
    assert reference.features["HasCrCard"].is_discrete
    assert not reference.features["Age"].is_discrete


def test_monitor_needs_minimum_samples(reference: ReferenceProfile, features: pd.DataFrame) -> None:
    monitor = make_monitor(reference, min_samples=100)
    monitor.observe(features.head(10), np.full(10, 0.2))
    assert monitor.report().status == DriftMonitor.STATUS_INSUFFICIENT


def test_no_drift_on_training_like_traffic(reference: ReferenceProfile, features: pd.DataFrame) -> None:
    monitor = make_monitor(reference)
    sample = features.sample(400, random_state=1)
    monitor.observe(sample, np.random.default_rng(1).uniform(size=len(sample)))
    report = monitor.report()
    assert report.status == DriftMonitor.STATUS_OK, report.drifted_features


def test_drift_detected_when_population_shifts(reference: ReferenceProfile, features: pd.DataFrame) -> None:
    monitor = make_monitor(reference)
    shifted = features.sample(400, random_state=2).assign(Age=lambda d: d["Age"] + 25, Geography="Germany")
    monitor.observe(shifted, np.random.default_rng(2).uniform(size=len(shifted)))
    report = monitor.report()
    assert report.drift_detected
    assert {"Age", "Geography"} <= set(report.drifted_features)


def test_request_metrics_snapshot() -> None:
    metrics = RequestMetrics(latency_window=100)
    for latency in (10.0, 20.0, 30.0):
        metrics.record("/predict", 200, latency)
    metrics.record("/predict", 422, 5.0)
    snap = metrics.snapshot()
    assert snap["requests_total"] == 4
    assert snap["errors_by_route"] == {"/predict": 1}
    assert snap["latency"]["avg_ms"] == pytest.approx(16.25)
