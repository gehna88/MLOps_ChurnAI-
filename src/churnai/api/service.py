"""Inference service: the boundary between HTTP and the model.

Routes stay thin; all model interaction (prediction, explanation, drift
recording) lives here so it can be unit-tested without HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from churnai.config import Settings
from churnai.explain.explainer import ChurnExplainer, Explanation
from churnai.models.bundle import ModelBundle
from churnai.monitoring.drift import DriftMonitor

PROBA_DECIMALS = 4


@dataclass(frozen=True)
class Prediction:
    churn_probability: float
    churn_prediction: bool
    threshold: float
    model_version: str
    model_variant: str = "A"


class ModelService:
    def __init__(self, bundle: ModelBundle, settings: Settings, variant: str = "A") -> None:
        self.bundle = bundle
        self.variant = variant
        self.explainer = ChurnExplainer(bundle, top_k=settings.explain.top_k)
        self.drift = DriftMonitor(
            reference=bundle.reference,
            window_size=settings.monitoring.drift_window_size,
            min_samples=settings.monitoring.drift_min_samples,
            alert_threshold=settings.monitoring.psi_alert_threshold,
        )

    @property
    def version(self) -> str:
        return self.bundle.metadata.version

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.bundle.predict_proba(X)
        self.drift.observe(X, proba)
        return proba

    def _to_predictions(self, proba: np.ndarray) -> list[Prediction]:
        labels = self.bundle.predict_label(proba)
        threshold = self.bundle.metadata.threshold
        return [
            Prediction(round(float(p), PROBA_DECIMALS), bool(lbl), threshold, self.version, self.variant)
            for p, lbl in zip(proba, labels, strict=True)
        ]

    def predict(self, X: pd.DataFrame) -> list[Prediction]:
        return self._to_predictions(self._score(X))

    def explain(self, X: pd.DataFrame) -> list[tuple[Prediction, Explanation]]:
        predictions = self.predict(X)
        return list(zip(predictions, self.explainer.explain(X), strict=True))
