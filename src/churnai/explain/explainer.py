"""SHAP explanations aggregated back to business-meaningful features.

The explainer is built once (at API start-up) and reused for every request.
Tree models use ``TreeExplainer``; linear models use ``LinearExplainer`` with
the bundle's background sample. Contributions are in log-odds units: positive
values push the customer towards churning, negative values away from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from churnai.features.preprocessing import FEATURES_STEP, PREPROCESS_STEP, source_features
from churnai.models.bundle import POSITIVE_CLASS_INDEX, ModelBundle

SHAP_UNITS = "log_odds"
INCREASES = "increases_churn_risk"
DECREASES = "decreases_churn_risk"
SHAP_DECIMALS = 4
# Keeps log-odds finite for probabilities of exactly 0 or 1.
PROBA_CLIP = 1e-7


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: Any
    shap_value: float

    @property
    def direction(self) -> str:
        return INCREASES if self.shap_value > 0 else DECREASES


@dataclass(frozen=True)
class Explanation:
    base_value: float
    contributions: list[FeatureContribution]
    units: str = SHAP_UNITS


def _is_linear(model: BaseEstimator) -> bool:
    return hasattr(model, "coef_")


def _positive_class(values: Any) -> np.ndarray:
    """Normalise the various shapes SHAP returns for binary classifiers."""
    if isinstance(values, list):
        return np.asarray(values[POSITIVE_CLASS_INDEX])
    values = np.asarray(values)
    if values.ndim == 3:  # noqa: PLR2004 - (rows, features, classes)
        return values[:, :, POSITIVE_CLASS_INDEX]
    return values


def _log_odds(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, PROBA_CLIP, 1 - PROBA_CLIP)
    return np.log(p / (1 - p))


class ChurnExplainer:
    def __init__(self, bundle: ModelBundle, top_k: int) -> None:
        self._setup(bundle.pipeline, bundle.metadata.feature_columns, bundle.background, top_k)

    @classmethod
    def from_pipeline(
        cls, pipeline: Pipeline, feature_columns: list[str], background: pd.DataFrame, top_k: int
    ) -> ChurnExplainer:
        """For models that are not in a bundle, such as every candidate in an MLflow run."""
        explainer = cls.__new__(cls)
        explainer._setup(pipeline, feature_columns, background, top_k)
        return explainer

    def _setup(
        self, pipeline: Pipeline, feature_columns: list[str], background: pd.DataFrame, top_k: int
    ) -> None:
        self.top_k = top_k
        self._engineer = pipeline.named_steps[FEATURES_STEP]
        self._preprocess = pipeline.named_steps[PREPROCESS_STEP]
        self._feature_columns = feature_columns
        self._sources = np.asarray(source_features(self._preprocess))

        self._model = pipeline[-1]
        if _is_linear(self._model):
            background = self._preprocess.transform(self._engineer.transform(background))
            self._explainer = shap.LinearExplainer(self._model, background)
        else:
            self._explainer = shap.TreeExplainer(self._model)

    def _base_values(self, transformed: np.ndarray, shap_values: np.ndarray) -> np.ndarray:
        """Base value implied by the model's own prediction.

        ``explainer.expected_value`` is not reliable across library versions
        (SHAP 0.46 reports 0 for XGBoost 2.x models). Deriving it from the
        model's log-odds guarantees ``base + sum(contributions) == prediction``.
        """
        proba = self._model.predict_proba(transformed)[:, POSITIVE_CLASS_INDEX]
        return _log_odds(proba) - shap_values.sum(axis=1)

    def _aggregate(self, shap_row: np.ndarray) -> pd.Series:
        """Sum one-hot contributions into their parent feature."""
        return pd.Series(shap_row).groupby(self._sources).sum()

    def contributions(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """SHAP values per customer and business feature, with those features' values."""
        engineered = self._engineer.transform(X[self._feature_columns])
        shap_values = _positive_class(self._explainer.shap_values(self._preprocess.transform(engineered)))
        by_feature = pd.DataFrame(shap_values, columns=self._sources).T.groupby(level=0).sum().T
        return by_feature, engineered.reset_index(drop=True)[list(by_feature.columns)]

    def explain(self, X: pd.DataFrame) -> list[Explanation]:
        engineered = self._engineer.transform(X[self._feature_columns])
        transformed = self._preprocess.transform(engineered)
        shap_values = _positive_class(self._explainer.shap_values(transformed))
        base_values = self._base_values(transformed, shap_values)

        explanations = []
        for row_idx in range(len(X)):
            aggregated = self._aggregate(shap_values[row_idx])
            top = aggregated.reindex(aggregated.abs().sort_values(ascending=False).index)
            contributions = [
                FeatureContribution(
                    feature=feature,
                    value=_to_builtin(engineered.iloc[row_idx][feature]),
                    shap_value=round(float(value), SHAP_DECIMALS),
                )
                for feature, value in top.head(self.top_k).items()
            ]
            base = round(float(base_values[row_idx]), SHAP_DECIMALS)
            explanations.append(Explanation(base, contributions))
        return explanations


def _to_builtin(value: Any) -> Any:
    """numpy scalars -> plain Python so they serialise cleanly to JSON."""
    return value.item() if isinstance(value, np.generic) else value
