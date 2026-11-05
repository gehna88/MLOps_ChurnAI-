"""Offline A/B: is the champion really better than each other candidate, or is it noise?

Run with ``make compare`` or ``python -m churnai.pipelines.compare_models``, after ``make train``.

* Repeated stratified CV on the training split, same folds for every model, compared
  with the corrected resampled t-test (Holm-adjusted). This decides the verdict.
* On the test split: a bootstrap 95% CI of the AUC difference and McNemar's test on the
  thresholded predictions. These confirm the verdict and never select a model.

Outputs: MLflow experiment ``churnai-ab`` (a parent run, a nested run per pair) and
``reports/ab_offline.csv`` / ``reports/ab_offline.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from churnai.config import Settings
from churnai.data import io
from churnai.data.schema import FEATURE_COLUMNS
from churnai.exceptions import ConfigError
from churnai.logging_utils import get_logger
from churnai.models.bundle import ModelBundle
from churnai.models.comparison import (
    bootstrap_auc_diff,
    corrected_resampled_ttest,
    holm_adjust,
    mcnemar_exact,
)
from churnai.models.evaluation import to_labels
from churnai.models.factory import ModelSpec, load_model_specs
from churnai.models.tracking import ExperimentTracker, make_tracker
from churnai.models.training import ModelTrainer, predict_positive
from churnai.pipelines._cli import exit_with, run_cli

logger = get_logger(__name__)

METRIC_DECIMALS = 4
VERDICT_A = "A_better"
VERDICT_B = "B_better"
VERDICT_TIE = "no_significant_difference"


@dataclass(frozen=True)
class Scored:
    """One model's paired CV fold scores and its held-out predictions."""

    name: str
    cv_scores: np.ndarray
    test_proba: np.ndarray
    test_labels: np.ndarray


def _score(
    spec: ModelSpec, trainer: ModelTrainer, repeats: int, X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> Scored:
    result = trainer.evaluate(spec, X_train, y_train, X_test, y_test)
    proba = predict_positive(result.pipeline, X_test)
    cv_scores = trainer.repeated_cv_auc(spec, X_train, y_train, repeats)
    return Scored(spec.name, cv_scores, proba, to_labels(proba, result.threshold))


def compare_pair(a: Scored, b: Scored, y_test: np.ndarray, settings: Settings) -> dict[str, float | str]:
    ab = settings.ab_test
    ttest = corrected_resampled_ttest(b.cv_scores - a.cv_scores, settings.training.cv_folds)
    interval = bootstrap_auc_diff(
        y_test, a.test_proba, b.test_proba, ab.bootstrap_iterations, settings.project.random_seed
    )
    mcnemar = mcnemar_exact(y_test, a.test_labels, b.test_labels)
    return {
        "model_a": a.name,
        "model_b": b.name,
        "cv_auc_a": float(a.cv_scores.mean()),
        "cv_auc_b": float(b.cv_scores.mean()),
        "cv_auc_diff": ttest.mean_diff,
        "t_stat": ttest.t_stat,
        "p_value": ttest.p_value,
        "test_auc_diff": interval.mean_diff,
        "test_auc_diff_ci_low": interval.ci_low,
        "test_auc_diff_ci_high": interval.ci_high,
        "mcnemar_a_only_correct": mcnemar.a_only_correct,
        "mcnemar_b_only_correct": mcnemar.b_only_correct,
        "mcnemar_p_value": mcnemar.p_value,
    }


def _verdict(row: pd.Series, alpha: float) -> str:
    if row["p_value_holm"] >= alpha:
        return VERDICT_TIE
    return VERDICT_B if row["cv_auc_diff"] > 0 else VERDICT_A


def _write_report(table: pd.DataFrame, settings: Settings) -> list:
    reports_dir = settings.resolve(settings.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "ab_offline.csv"
    md_path = reports_dir / "ab_offline.md"
    table.to_csv(csv_path, index=False)
    columns = ["model_a", "model_b", "cv_auc_a", "cv_auc_b", "p_value", "p_value_holm", "test_auc_diff",
               "test_auc_diff_ci_low", "test_auc_diff_ci_high", "mcnemar_p_value", "verdict"]
    md_path.write_text(
        "# Offline A/B comparison\n\n"
        f"A is the champion. Differences are B - A. {settings.ab_test.cv_repeats} x "
        f"{settings.training.cv_folds} repeated CV on the training split; test-split columns "
        "are confirmation only.\n\n"
        + table[columns].round(METRIC_DECIMALS).to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    return [csv_path, md_path]


def _log_pair(tracker: ExperimentTracker, row: pd.Series, a: Scored, b: Scored) -> None:
    with tracker.run(f"{a.name}_vs_{b.name}", nested=True, tags={"verdict": row["verdict"]}):
        tracker.log_params({"model_a": a.name, "model_b": b.name})
        metrics = row.drop(["model_a", "model_b", "verdict"]).astype(float).to_dict()
        tracker.log_metrics(metrics)
        tracker.log_dict(
            {a.name: a.cv_scores.tolist(), b.name: b.cv_scores.tolist()}, "cv_fold_scores.json"
        )


def run(settings: Settings, tracker: ExperimentTracker | None = None) -> pd.DataFrame:
    tracker = tracker or make_tracker(settings, settings.mlflow.ab_experiment)
    X_train, y_train = io.split_features_target(io.read_csv(settings.train_path))
    X_test, y_test = io.split_features_target(io.read_csv(settings.test_path))
    X_train, X_test = X_train[list(FEATURE_COLUMNS)], X_test[list(FEATURE_COLUMNS)]

    specs = load_model_specs(settings.resolve(settings.training.models_config))
    champion = ModelBundle.load(settings.model_path).metadata.model_name
    if champion not in {s.name for s in specs}:
        raise ConfigError(f"Champion '{champion}' is not in the candidates. Run `make train`.")

    trainer = ModelTrainer(
        settings.training.cv_folds, settings.project.random_seed, settings.training.threshold_metric
    )
    scored = {
        s.name: _score(s, trainer, settings.ab_test.cv_repeats, X_train, y_train, X_test, y_test)
        for s in specs
    }
    logger.info("Scored %d models on identical folds", len(scored))

    a = scored[champion]
    others = [scored[s.name] for s in specs if s.name != champion]
    table = pd.DataFrame([compare_pair(a, b, y_test.to_numpy(), settings) for b in others])
    table["p_value_holm"] = holm_adjust(table["p_value"].tolist())
    table["verdict"] = table.apply(_verdict, axis=1, alpha=settings.ab_test.alpha)
    table = table.sort_values("cv_auc_b", ascending=False).reset_index(drop=True)

    paths = _write_report(table, settings)
    with tracker.run(f"offline_ab_{champion}", tags={"type": "offline_ab"}):
        tracker.log_params({"champion": champion, "cv_repeats": settings.ab_test.cv_repeats,
                            "cv_folds": settings.training.cv_folds,
                            "bootstrap_iterations": settings.ab_test.bootstrap_iterations})
        for path in paths:
            tracker.log_artifact(path)
        for _, row in table.iterrows():
            _log_pair(tracker, row, a, scored[row["model_b"]])
    logger.info("Offline A/B results written to %s", paths[1])
    return table


def main() -> None:
    exit_with(run_cli("Compare the champion with every candidate using paired statistical tests.", run))


if __name__ == "__main__":
    main()
