"""Phase 2 pipeline: compare candidate models, track them, ship the champion.

Run with ``make train`` or ``python -m churnai.pipelines.train``.

Outputs
-------
* One MLflow run per candidate (params, CV + test metrics, confusion matrix, model, SHAP plots)
* The champion registered in the MLflow Model Registry with the ``champion`` alias,
  and the runner-up with the ``challenger`` alias (both with their bundles attached)
* ``models/model.joblib``  - the bundle the API serves (latest champion)
* ``models/<version>.joblib`` - immutable versioned copy, e.g. ``xgb_shallow-v2-65c7276a.joblib``
* ``models/metadata.json`` - human-readable lineage of that bundle
* ``reports/leaderboard.csv`` - side-by-side comparison of all candidates
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from churnai.config import Settings
from churnai.data import io
from churnai.data.schema import FEATURE_COLUMNS
from churnai.explain.report import SHAP_ARTIFACT_PATH, write_shap_report
from churnai.logging_utils import get_logger
from churnai.models.bundle import ModelBundle, ModelMetadata
from churnai.models.factory import load_model_specs
from churnai.models.tracking import ExperimentTracker, make_tracker
from churnai.models.training import (
    CandidateResult,
    ModelTrainer,
    guard_against_leakage,
    predict_positive,
    sample_background,
    select_challenger,
    select_champion,
)
from churnai.monitoring.drift import ReferenceProfile
from churnai.pipelines._cli import exit_with, run_cli

logger = get_logger(__name__)

INPUT_EXAMPLE_ROWS = 3


@dataclass(frozen=True)
class TrainingOutcome:
    champion: str
    model_version: str
    bundle_path: Path
    leaderboard: pd.DataFrame
    challenger: str | None = None


def _log_shap(
    settings: Settings, tracker: ExperimentTracker, result: CandidateResult, X_train: pd.DataFrame
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for path in write_shap_report(result.pipeline, X_train, settings, Path(tmp)):
            tracker.log_artifact(path, SHAP_ARTIFACT_PATH)


def _log_candidate(tracker: ExperimentTracker, result: CandidateResult, example: pd.DataFrame) -> None:
    tracker.log_params({"estimator": result.spec.estimator, "threshold": result.threshold,
                        **result.params})
    tracker.log_metrics(result.cv_metrics, prefix="cv_")
    tracker.log_metrics(result.test_metrics, prefix="test_")
    tracker.log_dict(result.test_confusion, "confusion_matrix.json")
    tracker.log_model(result.pipeline, example)


def build_bundle(
    settings: Settings, champion: CandidateResult, X_train: pd.DataFrame,
    run_id: str | None, registered_version: str | None,
) -> ModelBundle:
    reference = ReferenceProfile.fit(
        X_train, predict_positive(champion.pipeline, X_train), settings.monitoring.psi_bins
    )
    metadata = ModelMetadata(
        model_name=champion.name,
        estimator=champion.spec.estimator,
        threshold=champion.threshold,
        params=champion.params,
        cv_metrics=champion.cv_metrics,
        test_metrics=champion.test_metrics,
        mlflow_run_id=run_id,
        registered_version=registered_version,
    )
    background = sample_background(
        X_train, settings.explain.background_size, settings.project.random_seed
    )
    return ModelBundle(champion.pipeline, metadata, reference, background)


def _ship_challenger(
    settings: Settings, tracker: ExperimentTracker, challenger: CandidateResult,
    X_train: pd.DataFrame, run_id: str,
) -> None:
    """Register the runner-up and attach its bundle so the A/B test can load it by alias."""
    version = tracker.register(run_id, alias=settings.mlflow.challenger_alias)
    bundle = build_bundle(settings, challenger, X_train, run_id, version)
    tracker.log_bundle(run_id, bundle.save(settings.versioned_model_path(bundle.metadata.version)))


def run(settings: Settings, tracker: ExperimentTracker | None = None) -> TrainingOutcome:
    tracker = tracker or make_tracker(settings)
    X_train, y_train = io.split_features_target(io.read_csv(settings.train_path))
    X_test, y_test = io.split_features_target(io.read_csv(settings.test_path))
    X_train, X_test = X_train[list(FEATURE_COLUMNS)], X_test[list(FEATURE_COLUMNS)]

    trainer = ModelTrainer(
        cv_folds=settings.training.cv_folds,
        random_seed=settings.project.random_seed,
        threshold_metric=settings.training.threshold_metric,
    )
    # Float-typed example so MLflow's inferred signature tolerates missing values.
    example = X_train.head(INPUT_EXAMPLE_ROWS).astype(
        {c: "float64" for c in X_train.select_dtypes("number").columns}
    )

    results: list[CandidateResult] = []
    run_ids: dict[str, str | None] = {}
    for spec in load_model_specs(settings.resolve(settings.training.models_config)):
        with tracker.run(spec.name, tags={"estimator": spec.estimator}) as run_id:
            result = trainer.evaluate(spec, X_train, y_train, X_test, y_test)
            _log_candidate(tracker, result, example)
            _log_shap(settings, tracker, result, X_train)
        results.append(result)
        run_ids[spec.name] = run_id
        logger.info("Candidate %-16s cv_roc_auc=%.4f test_roc_auc=%.4f threshold=%.2f",
                    spec.name, result.cv_metrics["roc_auc"], result.test_metrics["roc_auc"],
                    result.threshold)

    champion = select_champion(results, settings.training.primary_metric)
    guard_against_leakage(champion, settings.training.leakage_auc_threshold)

    run_id = run_ids[champion.name]
    registered_version = tracker.register(run_id) if run_id else None
    bundle = build_bundle(settings, champion, X_train, run_id, registered_version)
    versioned_path = bundle.save(settings.versioned_model_path(bundle.metadata.version))
    if run_id:
        tracker.log_bundle(run_id, versioned_path)
    bundle_path = bundle.save(settings.model_path)
    io.write_json(bundle.metadata.to_dict(), settings.model_dir / "metadata.json")

    challenger = select_challenger(results, settings.training.primary_metric, champion)
    if challenger and run_ids[challenger.name]:
        _ship_challenger(settings, tracker, challenger, X_train, run_ids[challenger.name])

    leaderboard = pd.DataFrame([r.leaderboard_row() for r in results]).sort_values(
        f"cv_{settings.training.primary_metric}", ascending=False
    )
    reports_dir = settings.resolve(settings.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(reports_dir / "leaderboard.csv", index=False)

    logger.info("Champion: %s (%s) saved to %s", champion.name, bundle.metadata.version,
                bundle_path)
    return TrainingOutcome(
        champion.name, bundle.metadata.version, bundle_path, leaderboard,
        challenger.name if challenger else None,
    )


def main() -> None:
    exit_with(run_cli("Train candidate models and save the champion.", run))


if __name__ == "__main__":
    main()
