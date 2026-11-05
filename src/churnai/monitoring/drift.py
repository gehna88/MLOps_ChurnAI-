"""Population Stability Index (PSI) drift detection.

At training time a :class:`ReferenceProfile` captures the distribution of every
feature and of the predicted probability. At serving time a
:class:`DriftMonitor` keeps a rolling window of recent requests and compares it
to the reference. PSI above the alert threshold (0.2 by convention) flags drift.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Floor applied to bin proportions so empty bins don't produce log(0).
PSI_EPSILON = 1e-4
PREDICTION_KEY = "__prediction__"
PSI_DECIMALS = 4


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI between two aligned proportion vectors."""
    e = np.clip(np.asarray(expected, dtype=float), PSI_EPSILON, None)
    a = np.clip(np.asarray(actual, dtype=float), PSI_EPSILON, None)
    return float(np.sum((a - e) * np.log(a / e)))


@dataclass(frozen=True)
class FeatureReference:
    """Binned distribution of one feature in the training data."""

    name: str
    proportions: tuple[float, ...]
    cut_points: tuple[float, ...] = ()  # numeric: interior bin edges
    categories: tuple[Any, ...] = ()  # discrete: observed values

    @property
    def is_discrete(self) -> bool:
        return bool(self.categories)

    @classmethod
    def fit(cls, name: str, values: pd.Series, bins: int) -> FeatureReference:
        if values.dtype == object or values.nunique() <= bins:
            shares = values.value_counts(normalize=True).sort_index()
            return cls(name, tuple(shares.to_numpy()), categories=tuple(shares.index))
        quantiles = np.quantile(values, np.linspace(0, 1, bins + 1))
        cut_points = tuple(np.unique(quantiles)[1:-1])
        counts = np.bincount(np.digitize(values, cut_points), minlength=len(cut_points) + 1)
        return cls(name, tuple(counts / counts.sum()), cut_points=cut_points)

    def proportions_of(self, values: pd.Series) -> np.ndarray:
        if self.is_discrete:
            # Unseen categories are ignored here; they are rejected upstream by validation.
            return np.array([(values == c).mean() for c in self.categories])
        counts = np.bincount(np.digitize(values, self.cut_points), minlength=len(self.proportions))
        return counts / max(counts.sum(), 1)

    def psi_against(self, values: pd.Series) -> float:
        return round(psi(np.array(self.proportions), self.proportions_of(values)), PSI_DECIMALS)


@dataclass(frozen=True)
class ReferenceProfile:
    features: dict[str, FeatureReference]
    prediction: FeatureReference
    prediction_mean: float
    n_samples: int

    @classmethod
    def fit(cls, X: pd.DataFrame, proba: np.ndarray, bins: int) -> ReferenceProfile:
        features = {c: FeatureReference.fit(c, X[c], bins) for c in X.columns}
        prediction = FeatureReference.fit(PREDICTION_KEY, pd.Series(proba), bins)
        return cls(features, prediction, float(np.mean(proba)), len(X))


@dataclass
class DriftReport:
    status: str
    window_size: int
    feature_psi: dict[str, float] = field(default_factory=dict)
    drifted_features: list[str] = field(default_factory=list)
    prediction_psi: float | None = None
    prediction_mean_reference: float | None = None
    prediction_mean_live: float | None = None
    drift_detected: bool = False


class DriftMonitor:
    """Thread-safe rolling window of live inputs compared against a reference."""

    STATUS_INSUFFICIENT = "insufficient_data"
    STATUS_OK = "ok"
    STATUS_DRIFT = "drift_detected"

    def __init__(
        self,
        reference: ReferenceProfile,
        window_size: int,
        min_samples: int,
        alert_threshold: float,
    ) -> None:
        self.reference = reference
        self.min_samples = min_samples
        self.alert_threshold = alert_threshold
        self._rows: deque[dict[str, Any]] = deque(maxlen=window_size)
        self._predictions: deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def observe(self, features: pd.DataFrame, proba: np.ndarray) -> None:
        with self._lock:
            self._rows.extend(features.to_dict(orient="records"))
            self._predictions.extend(float(p) for p in proba)

    def report(self) -> DriftReport:
        with self._lock:
            rows = list(self._rows)
            predictions = np.array(self._predictions)

        if len(rows) < self.min_samples:
            return DriftReport(status=self.STATUS_INSUFFICIENT, window_size=len(rows))

        live = pd.DataFrame(rows)
        feature_psi = {
            name: ref.psi_against(live[name])
            for name, ref in self.reference.features.items()
            if name in live.columns
        }
        drifted = sorted(n for n, v in feature_psi.items() if v > self.alert_threshold)
        prediction_psi = self.reference.prediction.psi_against(pd.Series(predictions))
        drift = bool(drifted) or prediction_psi > self.alert_threshold
        return DriftReport(
            status=self.STATUS_DRIFT if drift else self.STATUS_OK,
            window_size=len(rows),
            feature_psi=feature_psi,
            drifted_features=drifted,
            prediction_psi=prediction_psi,
            prediction_mean_reference=round(self.reference.prediction_mean, PSI_DECIMALS),
            prediction_mean_live=round(float(predictions.mean()), PSI_DECIMALS),
            drift_detected=drift,
        )
