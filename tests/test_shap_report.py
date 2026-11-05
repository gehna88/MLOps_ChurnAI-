"""Global SHAP reports: their contents, logging during training, and the backfill."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from churnai.config import Settings
from churnai.data import io
from churnai.data.schema import FEATURE_COLUMNS
from churnai.explain.explainer import ChurnExplainer
from churnai.explain.report import (
    BAR_FILE,
    BEESWARM_FILE,
    IMPORTANCE_FILE,
    SHAP_ARTIFACT_PATH,
    write_shap_report,
)
from churnai.models.bundle import ModelBundle
from churnai.models.tracking import make_tracker
from churnai.pipelines import backfill_shap, prepare_data, train
from tests.conftest import make_project

pytestmark = pytest.mark.slow

REPORT_FILES = {IMPORTANCE_FILE, BAR_FILE, BEESWARM_FILE}
TWO_MODELS = {
    "candidates": [
        {"name": "logreg", "estimator": "logistic_regression", "params": {"max_iter": 500}},
        {"name": "lgbm_tiny", "estimator": "lightgbm", "params": {"n_estimators": 20}},
    ]
}


def training_features(settings: Settings) -> pd.DataFrame:
    X, _ = io.split_features_target(io.read_csv(settings.train_path))
    return X[list(FEATURE_COLUMNS)]


def shap_files(settings: Settings, run_id: str) -> set[str]:
    client = mlflow.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    return {Path(a.path).name for a in client.list_artifacts(run_id, SHAP_ARTIFACT_PATH)}


def test_report_ranks_features_by_mean_absolute_shap(
    trained_project: Settings, bundle: ModelBundle, tmp_path: Path
) -> None:
    X = training_features(trained_project)
    paths = write_shap_report(bundle.pipeline, X, trained_project, tmp_path)
    assert {p.name for p in paths} == REPORT_FILES
    assert all(p.stat().st_size > 0 for p in paths)

    importance = pd.read_csv(tmp_path / IMPORTANCE_FILE)
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert "Complain" not in set(importance["feature"])  # the leaky column never reaches the model
    assert importance["mean_abs_shap"].iloc[0] > 0


def test_contributions_agree_with_per_customer_explanations(
    trained_project: Settings, bundle: ModelBundle
) -> None:
    rows = training_features(trained_project).head(5)
    explainer = ChurnExplainer(bundle, top_k=3)
    values, features = explainer.contributions(rows)
    assert list(values.columns) == list(features.columns)
    assert len(values) == len(features) == len(rows)
    for row, explanation in enumerate(explainer.explain(rows)):
        for item in explanation.contributions:
            assert values[item.feature].iloc[row] == pytest.approx(item.shap_value, abs=1e-3)


def test_training_logs_a_report_and_backfill_fills_only_older_runs(tmp_path, sample_raw_df) -> None:
    settings = make_project(tmp_path, sample_raw_df, models=TWO_MODELS)
    prepare_data.run(settings)
    train.run(settings)

    tracker = make_tracker(settings)
    trained = tracker.model_run_ids()
    assert len(trained) == 2
    assert all(shap_files(settings, r) == REPORT_FILES for r in trained)

    # A run from before SHAP logging existed: a model, but no report.
    pipeline = tracker.load_pipeline(trained[0])
    example = training_features(settings).head(3).astype(
        {c: "float64" for c in training_features(settings).select_dtypes("number").columns}
    )
    with tracker.run("older_run") as older:
        tracker.log_params({"marker": "older"})
        tracker.log_metrics({"cv_roc_auc": 0.5})
        tracker.log_model(pipeline, example)
    assert shap_files(settings, older) == set()

    summary = backfill_shap.run(settings)
    assert summary.updated == [older]
    assert set(summary.skipped) == set(trained)
    assert shap_files(settings, older) == REPORT_FILES

    client = mlflow.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    after = client.get_run(older)
    assert after.data.metrics == {"cv_roc_auc": 0.5}
    assert after.data.params == {"marker": "older"}
    assert after.info.status == "FINISHED"

    assert backfill_shap.run(settings).updated == []  # safe to run again


def test_shap_values_cover_every_customer(trained_project: Settings, bundle: ModelBundle) -> None:
    rows = training_features(trained_project).head(20)
    values, _ = ChurnExplainer(bundle, top_k=3).contributions(rows)
    assert np.isfinite(values.to_numpy()).all()
