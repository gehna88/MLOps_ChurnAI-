"""End-to-end training against a real (temporary, SQLite-backed) MLflow store."""

import mlflow
import pytest

from churnai.models.bundle import ModelBundle
from churnai.pipelines import prepare_data, train
from tests.conftest import make_project

pytestmark = pytest.mark.slow

FAST_SINGLE_MODEL = {
    "candidates": [
        {"name": "logreg", "estimator": "logistic_regression", "params": {"max_iter": 500}},
        {"name": "xgb_tiny", "estimator": "xgboost", "params": {"n_estimators": 20}},
    ]
}


def test_training_logs_runs_and_registers_champion(tmp_path, sample_raw_df) -> None:
    settings = make_project(tmp_path, sample_raw_df, models=FAST_SINGLE_MODEL)
    prepare_data.run(settings)
    outcome = train.run(settings)

    client = mlflow.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(settings.mlflow.experiment_name)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == len(FAST_SINGLE_MODEL["candidates"])
    assert all("cv_roc_auc" in r.data.metrics and "test_f1" in r.data.metrics for r in runs)

    champion = client.get_model_version_by_alias(
        settings.mlflow.registered_model_name, settings.mlflow.champion_alias
    )
    bundle = ModelBundle.load(outcome.bundle_path)
    assert bundle.metadata.mlflow_run_id == champion.run_id
    assert bundle.metadata.registered_version == str(champion.version)
    assert f"-v{champion.version}-" in outcome.model_version
