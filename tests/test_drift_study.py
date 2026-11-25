"""Sequential drift study: slicing, drift ordering, the full study, outputs and MLflow."""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from churnai.config import DriftScenario, Settings, load_settings
from churnai.exceptions import ConfigError
from churnai.models.tracking import NullTracker
from churnai.monitoring import drift_study as core
from churnai.pipelines import drift_study
from tests.conftest import CONFIG_FILE

N_SLICES = 4
FOCUS = ["Age", "Balance", "CreditScore", "EstimatedSalary"]
DRIFT = DriftScenario(name="drift", description="test", mean_shift=0.6, spread=0.6, noise=1.5)
NATURAL = DriftScenario(name="natural", description="test")


def _with_study(settings: Settings, **changes) -> Settings:
    return settings.model_copy(update={"drift_study": settings.drift_study.model_copy(update=changes)})


# ------------------------------------------------------------- slicing ---
def test_make_slices_partitions_every_row_once() -> None:
    df = pd.DataFrame({"x": range(10)})
    slices = core.make_slices(df, 3)
    assert [len(s) for s in slices] == [3, 4, 3]
    assert pd.concat(slices)["x"].tolist() == list(range(10))


def test_make_slices_rejects_too_few_slices_or_rows() -> None:
    with pytest.raises(ValueError, match="at least two"):
        core.make_slices(pd.DataFrame({"x": range(10)}), 1)
    with pytest.raises(ValueError, match="Cannot cut"):
        core.make_slices(pd.DataFrame({"x": range(3)}), 5)


def test_study_data_is_validated_and_in_row_order(project: Settings, sample_raw_df: pd.DataFrame) -> None:
    data = core.load_study_data(project)
    assert "Complain" not in data.columns and "RowNumber" not in data.columns
    expected = sample_raw_df.sort_values("RowNumber")["CreditScore"].tolist()
    assert data["CreditScore"].tolist() == expected


def test_natural_order_is_unchanged(project: Settings) -> None:
    data = core.load_study_data(project)
    pd.testing.assert_frame_equal(core.order_rows(data, NATURAL, FOCUS, seed=1), data)


def test_drift_order_uses_every_row_once_and_is_reproducible(project: Settings) -> None:
    data = core.load_study_data(project)
    ordered = core.order_rows(data, DRIFT, FOCUS, seed=1)
    assert sorted(ordered["Age"]) == sorted(data["Age"])
    assert len(ordered) == len(data)
    pd.testing.assert_frame_equal(ordered, core.order_rows(data, DRIFT, FOCUS, seed=1))


def test_drift_order_makes_later_slices_higher_and_more_varied(project: Settings) -> None:
    data = core.load_study_data(project)
    natural = core.make_slices(core.order_rows(data, NATURAL, FOCUS, 1), N_SLICES)
    drifted = core.make_slices(core.order_rows(data, DRIFT, FOCUS, 1), N_SLICES)
    for feature in FOCUS:
        assert drifted[-1][feature].mean() > drifted[0][feature].mean()
        psi_drift = core.feature_psi(drifted[0], drifted[-1], [feature], bins=10)[feature]
        psi_natural = core.feature_psi(natural[0], natural[-1], [feature], bins=10)[feature]
        assert psi_drift > psi_natural
    assert drifted[-1]["Age"].std() > drifted[0]["Age"].std()


# ------------------------------------------------------------ measures ---
def test_psi_levels() -> None:
    assert core.psi_level(0.05, 0.1, 0.2) == core.LEVEL_STABLE
    assert core.psi_level(0.15, 0.1, 0.2) == core.LEVEL_MODERATE
    assert core.psi_level(0.25, 0.1, 0.2) == core.LEVEL_SIGNIFICANT


def test_ks_detects_a_shift() -> None:
    rng = np.random.default_rng(0)
    same = pd.DataFrame({"x": rng.normal(size=500)})
    shifted = pd.DataFrame({"x": rng.normal(loc=1.0, size=500)})
    assert core.ks_tests(same, shifted, ["x"])["x"]["pvalue"] < 0.001
    assert core.ks_tests(same, same, ["x"])["x"]["statistic"] == 0


# -------------------------------------------------------------- config ---
def test_drift_features_must_be_numeric(tmp_path) -> None:
    raw = yaml.safe_load(CONFIG_FILE.read_text())
    raw["drift_study"]["drift_features"] = ["Geography"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="numeric"):
        load_settings(config_path=bad, root=tmp_path)


def test_unknown_model_name_is_a_config_error(project: Settings) -> None:
    with pytest.raises(ConfigError, match="not a candidate"):
        core.resolve_model_spec(_with_study(project, model="nope"))


def test_champion_needs_a_trained_model(project: Settings) -> None:
    with pytest.raises(ConfigError, match="make train"):
        core.resolve_model_spec(project)


def test_named_candidate_is_resolved_from_the_catalogue(project: Settings) -> None:
    assert core.resolve_model_spec(_with_study(project, model="logreg")).estimator == "logistic_regression"


@pytest.mark.slow
def test_slice_count_is_validated(trained_project: Settings) -> None:
    with pytest.raises(ConfigError, match="between"):
        drift_study.run(trained_project, tracker=NullTracker(), n_slices=50)
    with pytest.raises(ConfigError, match="Unknown drift scenario"):
        drift_study.run(trained_project, tracker=NullTracker(), scenarios=["nope"])


# --------------------------------------------------------- full study ---
@pytest.fixture(scope="module")
def study(trained_project: Settings) -> drift_study.DriftStudyOutcome:
    return drift_study.run(trained_project, tracker=NullTracker(), n_slices=N_SLICES)


@pytest.mark.slow
def test_one_step_per_new_slice(study: drift_study.DriftStudyOutcome) -> None:
    for scenario in study.results["scenarios"]:
        steps = scenario["steps"]
        assert [s["step"] for s in steps] == list(range(1, N_SLICES))
        assert [s["new_slice"] for s in steps] == list(range(2, N_SLICES + 1))
        train_rows = [s["train_rows"] for s in steps]
        assert train_rows == sorted(train_rows) and len(set(train_rows)) == len(train_rows)
        assert len(scenario["slices"]) == N_SLICES


@pytest.mark.slow
def test_every_step_has_its_graphs(study: drift_study.DriftStudyOutcome) -> None:
    out = study.output_dir
    for scenario in study.results["scenarios"]:
        for step in scenario["steps"]:
            assert len(step["plots"]) == 2
            assert all((out / p).is_file() and (out / p).stat().st_size > 0 for p in step["plots"])
        assert len(scenario["plots"]) == len(drift_study.SUMMARY_PLOTS)
        assert all((out / p).is_file() for p in scenario["plots"])


@pytest.mark.slow
def test_results_report_and_comparison_are_written(study: drift_study.DriftStudyOutcome) -> None:
    out = study.output_dir
    assert json.loads((out / core.RESULTS_FILE).read_text()) == study.results
    report = (out / core.REPORT_FILE).read_text()
    assert "## Scenario: natural" in report and "## Scenario: gradual_drift" in report
    assert (out / core.COMPARISON_PLOT).is_file()
    assert not list(out.parent.glob(f".{out.name}*"))  # no staging or backup left behind


@pytest.mark.slow
def test_simulated_drift_beats_the_natural_control(study: drift_study.DriftStudyOutcome) -> None:
    last = {s["name"]: s["steps"][-1] for s in study.results["scenarios"]}
    worst = {name: max(step["psi_vs_initial"][f] for f in FOCUS) for name, step in last.items()}
    assert worst["gradual_drift"] > worst["natural"]
    assert worst["natural"] < 0.2


@pytest.mark.slow
def test_stale_model_equals_retrained_at_step_one(study: drift_study.DriftStudyOutcome) -> None:
    for scenario in study.results["scenarios"]:
        first = scenario["steps"][0]
        assert first["stale_metrics"] == first["retrained_metrics"]


@pytest.mark.slow
def test_step_metric_names_are_mlflow_safe(study: drift_study.DriftStudyOutcome) -> None:
    step = core.StepResult(**{k: v for k, v in study.results["scenarios"][0]["steps"][0].items()})
    names = drift_study.step_metrics(step)
    assert all(" " not in n for n in names)
    assert "psi_initial_Satisfaction_Score" in names and "retrained_roc_auc" in names


@pytest.mark.slow
def test_a_failed_study_keeps_the_previous_results(
    trained_project: Settings, study: drift_study.DriftStudyOutcome, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = (study.output_dir / core.RESULTS_FILE).read_text()

    def boom(*args, **kwargs):
        raise RuntimeError("plotting failed")

    monkeypatch.setattr(drift_study.plots, "plot_scenario_comparison", boom)
    with pytest.raises(RuntimeError, match="plotting failed"):
        drift_study.run(trained_project, tracker=NullTracker(), n_slices=3, scenarios=["natural"])
    assert (study.output_dir / core.RESULTS_FILE).read_text() == before
    assert not list(study.output_dir.parent.glob(f".{study.output_dir.name}*"))


@pytest.mark.slow
def test_cli_runs_with_options(trained_project: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHURNAI_HOME", str(trained_project.root))
    with pytest.raises(SystemExit) as exit_info:
        drift_study.main(["--slices", "3", "--scenario", "natural", "--no-mlflow"])
    assert exit_info.value.code == 0
    results = json.loads((trained_project.drift_study_dir / core.RESULTS_FILE).read_text())
    assert results["n_slices"] == 3 and [s["name"] for s in results["scenarios"]] == ["natural"]


# ---------------------------------------------------------------- MLflow ---
@pytest.mark.slow
def test_study_is_logged_to_the_data_drift_experiment(trained_project: Settings) -> None:
    outcome = drift_study.run(trained_project, n_slices=3)  # real MLflow tracker, temp SQLite store
    client = mlflow.MlflowClient(tracking_uri=trained_project.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name("data-drift")
    assert experiment is not None
    study_run = outcome.results["mlflow"]["run_id"]
    runs = client.search_runs([experiment.experiment_id])
    children = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == study_run]
    assert sorted(r.info.run_name for r in children) == ["gradual_drift", "natural"]

    scenario_run = outcome.results["scenarios"][1]["mlflow_run_id"]
    steps = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == scenario_run]
    assert len(steps) == 2
    assert all("psi_initial_Age" in r.data.metrics for r in steps)
    history = client.get_metric_history(scenario_run, "psi_initial_Age")
    assert sorted(m.step for m in history) == [1, 2]  # one point per iteration: a line in MLflow
    artifacts = {a.path for a in client.list_artifacts(steps[0].info.run_id, "plots")}
    assert any(p.endswith("_distributions.png") for p in artifacts)


def test_publish_works_when_the_old_study_cannot_be_renamed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside Docker the shipped study lives in an image layer, where renaming it fails (EXDEV)."""
    import errno
    from pathlib import Path

    final, staging = tmp_path / "drift_study", tmp_path / ".drift_study-new"
    final.mkdir()
    (final / "results.json").write_text("old")
    staging.mkdir()
    (staging / "results.json").write_text("new")
    real_rename = Path.rename

    def rename(self: Path, target):
        if self == final:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    drift_study._publish(staging, final)
    assert (final / "results.json").read_text() == "new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["drift_study"]
