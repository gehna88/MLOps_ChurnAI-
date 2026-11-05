"""The model bundle: the single hand-off artefact from training to serving.

A bundle contains everything the API needs and nothing it does not: the fitted
pipeline, the tuned decision threshold, lineage metadata linking it to an
MLflow run, the drift reference profile and a background sample for SHAP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churnai.data.schema import FEATURE_COLUMNS
from churnai.exceptions import ModelNotLoadedError
from churnai.monitoring.drift import ReferenceProfile

BUNDLE_FORMAT_VERSION = 1
POSITIVE_CLASS_INDEX = 1


@dataclass(frozen=True)
class ModelMetadata:
    model_name: str
    estimator: str
    threshold: float
    params: dict[str, Any]
    cv_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    mlflow_run_id: str | None = None
    registered_version: str | None = None
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    format_version: int = BUNDLE_FORMAT_VERSION

    @property
    def version(self) -> str:
        """Human-readable version tying a deployment back to its MLflow run."""
        registry = f"v{self.registered_version}" if self.registered_version else "local"
        run = (self.mlflow_run_id or "norun")[:8]
        return f"{self.model_name}-{registry}-{run}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "version": self.version}


@dataclass
class ModelBundle:
    pipeline: Pipeline
    metadata: ModelMetadata
    reference: ReferenceProfile
    background: pd.DataFrame

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.metadata.feature_columns])[:, POSITIVE_CLASS_INDEX]

    def predict_label(self, proba: np.ndarray) -> np.ndarray:
        return (proba >= self.metadata.threshold).astype(int)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        if not path.is_file():
            raise ModelNotLoadedError(f"Model artefact not found at {path}. Run `make train`.")
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise ModelNotLoadedError(f"File at {path} is not a ModelBundle")
        if bundle.metadata.format_version != BUNDLE_FORMAT_VERSION:
            raise ModelNotLoadedError(
                f"Bundle format v{bundle.metadata.format_version} is incompatible with "
                f"this code (expects v{BUNDLE_FORMAT_VERSION}). Retrain the model."
            )
        return bundle
