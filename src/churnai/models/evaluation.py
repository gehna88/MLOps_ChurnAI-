"""Classification metrics and decision-threshold tuning."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
METRIC_DECIMALS = 4

LabelMetric = Callable[[np.ndarray, np.ndarray], float]
LABEL_METRICS: dict[str, LabelMetric] = {
    "f1": lambda y, p: f1_score(y, p, zero_division=0),
    "recall": lambda y, p: recall_score(y, p, zero_division=0),
    "precision": lambda y, p: precision_score(y, p, zero_division=0),
    "accuracy": accuracy_score,
}


def to_labels(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(int)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    """Ranking metrics (threshold-free) plus label metrics at ``threshold``."""
    labels = to_labels(proba, threshold)
    metrics = {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        **{name: fn(y_true, labels) for name, fn in LABEL_METRICS.items()},
    }
    return {k: round(float(v), METRIC_DECIMALS) for k, v in metrics.items()}


def confusion(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, to_labels(proba, threshold), labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def tune_threshold(y_true: np.ndarray, proba: np.ndarray, metric: str = "f1") -> float:
    """Pick the threshold maximising ``metric``; tune on out-of-fold data only."""
    if metric not in LABEL_METRICS:
        raise ValueError(f"Unsupported threshold metric '{metric}'. Use {sorted(LABEL_METRICS)}")
    scorer = LABEL_METRICS[metric]
    scores = [scorer(y_true, to_labels(proba, t)) for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmax(scores))])
