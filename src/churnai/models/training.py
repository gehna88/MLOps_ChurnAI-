"""Candidate evaluation and champion selection.

Protocol (avoids optimistic bias):

1. Stratified K-fold cross-validation on the *training* split produces
   out-of-fold (OOF) probabilities.
2. Model selection uses the OOF primary metric, and the decision threshold is
   tuned on the same OOF probabilities - the test set is never used for either.
3. The candidate is refitted on the full training split and scored once on the
   held-out test split for an unbiased estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from churnai.exceptions import LeakageSuspectedError
from churnai.features.preprocessing import build_pipeline
from churnai.models.bundle import POSITIVE_CLASS_INDEX
from churnai.models.evaluation import compute_metrics, confusion, tune_threshold
from churnai.models.factory import BuildContext, ModelSpec, build_estimator, resolved_params


@dataclass
class CandidateResult:
    spec: ModelSpec
    pipeline: Pipeline
    params: dict[str, Any]
    threshold: float
    cv_metrics: dict[str, float]
    test_metrics: dict[str, float]
    test_confusion: dict[str, int]

    @property
    def name(self) -> str:
        return self.spec.name

    def score(self, metric: str) -> float:
        return self.cv_metrics[metric]

    def leaderboard_row(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "estimator": self.spec.estimator,
            "threshold": self.threshold,
            **{f"cv_{k}": v for k, v in self.cv_metrics.items()},
            **{f"test_{k}": v for k, v in self.test_metrics.items()},
        }


class ModelTrainer:
    def __init__(self, cv_folds: int, random_seed: int, threshold_metric: str) -> None:
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.threshold_metric = threshold_metric

    def _cv(self) -> StratifiedKFold:
        return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)

    def evaluate(
        self,
        spec: ModelSpec,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> CandidateResult:
        ctx = BuildContext.from_target(y_train.to_numpy(), self.random_seed)
        pipeline = build_pipeline(build_estimator(spec, ctx))

        oof = cross_val_predict(
            clone(pipeline), X_train, y_train, cv=self._cv(), method="predict_proba"
        )[:, POSITIVE_CLASS_INDEX]
        threshold = tune_threshold(y_train.to_numpy(), oof, self.threshold_metric)

        pipeline.fit(X_train, y_train)
        test_proba = pipeline.predict_proba(X_test)[:, POSITIVE_CLASS_INDEX]

        return CandidateResult(
            spec=spec,
            pipeline=pipeline,
            params=resolved_params(spec, ctx),
            threshold=threshold,
            cv_metrics=compute_metrics(y_train.to_numpy(), oof, threshold),
            test_metrics=compute_metrics(y_test.to_numpy(), test_proba, threshold),
            test_confusion=confusion(y_test.to_numpy(), test_proba, threshold),
        )


def select_champion(results: list[CandidateResult], metric: str) -> CandidateResult:
    if not results:
        raise ValueError("No candidate results to select from")
    return max(results, key=lambda r: r.score(metric))


def guard_against_leakage(result: CandidateResult, auc_threshold: float) -> None:
    """Implausibly good held-out ROC-AUC on this problem means leakage, not genius."""
    auc = result.test_metrics["roc_auc"]
    if auc > auc_threshold:
        raise LeakageSuspectedError(
            f"Test ROC-AUC {auc:.4f} for '{result.name}' exceeds {auc_threshold}. "
            "Check for target leakage before deploying."
        )


def sample_background(X: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Representative sample used by SHAP explainers that need background data."""
    return X.sample(n=min(size, len(X)), random_state=seed).reset_index(drop=True)


def predict_positive(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(X)[:, POSITIVE_CLASS_INDEX]
