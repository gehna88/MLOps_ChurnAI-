"""Tuning and A/B pipelines end to end, against a temporary SQLite-backed MLflow store."""

import mlflow
import pandas as pd
import pytest
import yaml

from churnai.api.ab import VARIANT_A, VARIANT_B
from churnai.config import Settings
from churnai.models.bundle import ModelBundle
from churnai.models.factory import load_model_specs
from churnai.pipelines import ab_online, compare_models, fetch_ab_bundles, prepare_data, train, tune
from tests.conftest import make_project

pytestmark = pytest.mark.slow

TINY_TUNING = {
    "n_trials": 2,
    "studies": [{
        "name": "xgb_tuned",
        "estimator": "xgboost",
        "fixed": {"scale_pos_weight": "auto"},
        "space": {
            "n_estimators": {"type": "int", "low": 10, "high": 30},
            "learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True},
        },
    }],
}


@pytest.fixture(scope="module")
def ab_project(tmp_path_factory: pytest.TempPathFactory, sample_raw_df: pd.DataFrame) -> Settings:
    """A trained project with fewer resamples, so the statistics run in seconds."""
    settings = make_project(tmp_path_factory.mktemp("ab"), sample_raw_df)
    settings = settings.model_copy(update={
        "ab_test": settings.ab_test.model_copy(update={"cv_repeats": 2, "bootstrap_iterations": 100}),
    })
    prepare_data.run(settings)
    train.run(settings)
    return settings


def client_for(settings: Settings) -> mlflow.MlflowClient:
    return mlflow.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def runs_in(settings: Settings, experiment: str) -> list:
    client = client_for(settings)
    return client.search_runs([client.get_experiment_by_name(experiment).experiment_id])


def test_tune_logs_nested_runs_and_writes_best_params_back(tmp_path, sample_raw_df) -> None:
    settings = make_project(tmp_path, sample_raw_df)
    models_path = settings.resolve(settings.training.models_config)
    models_path.write_text("# keep this header\n" + models_path.read_text())
    settings.resolve(settings.training.tuning_config).write_text(yaml.safe_dump(TINY_TUNING))
    prepare_data.run(settings)
    settings.test_path.unlink()  # tuning must not need the test split

    tune.run(settings)
    tune.run(settings)  # re-running replaces the candidate rather than duplicating it

    runs = runs_in(settings, settings.mlflow.tuning_experiment)
    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    children = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    assert len(parents) == 2
    assert len(children) == 4
    assert {r.data.tags["mlflow.parentRunId"] for r in children} == {r.info.run_id for r in parents}
    assert all("cv_roc_auc" in r.data.metrics and "n_estimators" in r.data.params for r in children)

    text = models_path.read_text()
    assert text.startswith("# keep this header")
    assert text.count(tune.TUNED_MARKER) == 1
    specs = {s.name: s for s in load_model_specs(models_path)}
    assert set(specs) == {"logreg", "xgb_small", "lgbm_small", "xgb_tuned"}
    assert specs["xgb_tuned"].params["scale_pos_weight"] == "auto"
    assert 10 <= specs["xgb_tuned"].params["n_estimators"] <= 30


def test_train_registers_champion_and_challenger_with_bundles(ab_project: Settings) -> None:
    client = client_for(ab_project)
    name = ab_project.mlflow.registered_model_name
    champion = client.get_model_version_by_alias(name, ab_project.mlflow.champion_alias)
    challenger = client.get_model_version_by_alias(name, ab_project.mlflow.challenger_alias)
    assert champion.run_id != challenger.run_id

    paths = fetch_ab_bundles.run(ab_project)
    a, b = ModelBundle.load(paths[VARIANT_A]), ModelBundle.load(paths[VARIANT_B])
    assert a.metadata.mlflow_run_id == champion.run_id
    assert b.metadata.mlflow_run_id == challenger.run_id
    assert a.metadata.cv_metrics["roc_auc"] >= b.metadata.cv_metrics["roc_auc"]


def test_compare_models_logs_each_pair_to_mlflow(ab_project: Settings) -> None:
    table = compare_models.run(ab_project)
    champion = ModelBundle.load(ab_project.model_path).metadata.model_name
    assert set(table["model_a"]) == {champion}
    assert len(table) == 2
    assert (table["p_value_holm"] >= table["p_value"]).all()
    assert (table["test_auc_diff_ci_low"] <= table["test_auc_diff"]).all()
    assert (table["test_auc_diff"] <= table["test_auc_diff_ci_high"]).all()

    runs = runs_in(ab_project, ab_project.mlflow.ab_experiment)
    pairs = [r for r in runs if "mlflow.parentRunId" in r.data.tags and "p_value_holm" in r.data.metrics]
    assert len(pairs) >= 2
    assert all(r.data.tags["verdict"] for r in pairs)
    assert (ab_project.resolve(ab_project.paths.reports_dir) / "ab_offline.md").is_file()


def test_online_replay_splits_customers_and_refuses_an_underpowered_verdict(ab_project: Settings) -> None:
    result = ab_online.run(ab_project)
    a, b = result["a"], result["b"]
    test_rows = len(pd.read_csv(ab_project.test_path))
    assert a.customers + b.customers == test_rows
    assert 0.4 < a.customers / test_rows < 0.6
    assert a.churners + b.churners == pd.read_csv(ab_project.test_path)["Exited"].sum()
    assert result["verdict"] == ab_online.VERDICT_UNDERPOWERED  # 500 test rows cannot detect 5 points

    runs = [r for r in runs_in(ab_project, ab_project.mlflow.ab_experiment)
            if r.data.tags.get("type") == "online_ab"]
    assert runs
    assert runs[0].data.metrics["a_customers"] == a.customers
    assert runs[0].data.tags["verdict"] == result["verdict"]


def test_online_replay_detects_misaligned_test_split(ab_project: Settings) -> None:
    project = ab_project.project.model_copy(update={"random_seed": 1})
    other = ab_project.model_copy(update={"project": project})
    with pytest.raises(ab_online.DataValidationError):
        ab_online.traffic_with_ids(other)


def _verdict_inputs(settings: Settings, b_catches: int) -> tuple[pd.DataFrame, ab_online.PowerPlan]:
    def arm(variant: str, catches: int) -> pd.DataFrame:
        churners, stayers = 1000, 1000
        return pd.DataFrame({
            "model_variant": variant,
            "model_version": variant,
            "exited": [1] * churners + [0] * stayers,
            "churn_prediction": [True] * catches + [False] * (churners - catches + stayers),
            "churn_probability": [0.9] * catches + [0.1] * (churners - catches + stayers),
        })
    outcomes = pd.concat([arm(VARIANT_A, 600), arm(VARIANT_B, b_catches)], ignore_index=True)
    return outcomes, ab_online.PowerPlan(0.6, 0.05, 900, 4500)


def test_powered_verdict_promotes_only_a_clear_winner(ab_project: Settings) -> None:
    outcomes, plan = _verdict_inputs(ab_project, b_catches=750)
    assert ab_online.analyze(outcomes, plan, ab_project)["verdict"] == ab_online.VERDICT_PROMOTE
    outcomes, plan = _verdict_inputs(ab_project, b_catches=605)
    assert ab_online.analyze(outcomes, plan, ab_project)["verdict"] == ab_online.VERDICT_KEEP
