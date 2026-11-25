"""Drift study API: read the study, serve graphs, run a study, compare customer sets."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churnai.api.app import create_app
from churnai.config import Settings
from churnai.data.schema import FEATURES
from churnai.models.bundle import ModelBundle
from churnai.models.tracking import NullTracker
from churnai.pipelines import drift_study, prepare_data, train
from tests.conftest import make_project

API_NAMES = {f.column: f.api_name for f in FEATURES}
STUDY_TIMEOUT_SECONDS = 300


def customers(df: pd.DataFrame) -> list[dict]:
    return df[list(API_NAMES)].rename(columns=API_NAMES).to_dict(orient="records")


@pytest.fixture(scope="module")
def studied(trained_project: Settings) -> Settings:
    drift_study.run(trained_project, tracker=NullTracker(), n_slices=4)
    return trained_project


@pytest.fixture
def client(studied: Settings, bundle: ModelBundle) -> Iterator[TestClient]:
    with TestClient(create_app(studied, bundle)) as test_client:
        yield test_client


@pytest.fixture
def empty_client(tmp_path, sample_raw_df: pd.DataFrame) -> Iterator[TestClient]:
    """No model, no study."""
    with TestClient(create_app(make_project(tmp_path, sample_raw_df))) as test_client:
        yield test_client


# ----------------------------------------------------------------- read ---
def test_no_study_yet_is_404(empty_client: TestClient) -> None:
    for path in ("/drift/study", "/drift/study/report", "/drift/study/plots",
                 "/drift/study/files/scenario_comparison.png"):
        assert empty_client.get(path).status_code == 404
    assert empty_client.get("/drift/study/status").json()["status"] == "idle"


@pytest.mark.slow
def test_study_results(client: TestClient) -> None:
    body = client.get("/drift/study").json()
    assert body["n_slices"] == 4
    assert [s["name"] for s in body["scenarios"]] == ["natural", "gradual_drift"]


@pytest.mark.slow
def test_scenario_and_step(client: TestClient) -> None:
    scenario = client.get("/drift/study/scenarios/gradual_drift").json()
    assert len(scenario["steps"]) == 3
    step = client.get("/drift/study/scenarios/gradual_drift/steps/2").json()
    assert step["new_slice"] == 3 and "Age" in step["psi_vs_initial"]
    assert client.get("/drift/study/scenarios/nope").status_code == 404
    assert client.get("/drift/study/scenarios/natural/steps/9").status_code == 404


@pytest.mark.slow
def test_every_listed_graph_is_served_as_png(client: TestClient) -> None:
    plots = client.get("/drift/study/plots").json()
    urls = [plots["comparison"]]
    for scenario in plots["scenarios"].values():
        urls += scenario["summary"] + [u for step in scenario["steps"].values() for u in step]
    assert len(urls) == 1 + 2 * (5 + 3 * 2)
    for url in urls:
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:4] == b"\x89PNG"


@pytest.mark.slow
def test_only_study_graphs_are_served(client: TestClient) -> None:
    for path in ("results.json", "../models/model.joblib", "%2e%2e/configs/config.yaml",
                 "natural/../results.json"):
        assert client.get(f"/drift/study/files/{path}").status_code == 404


@pytest.mark.slow
def test_report_is_markdown(client: TestClient) -> None:
    response = client.get("/drift/study/report")
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text.startswith("# Data drift study")


@pytest.mark.slow
def test_home_links_to_the_drift_study(client: TestClient) -> None:
    assert client.get("/").json()["drift_study"] == "/drift/study"


# ------------------------------------------------------------------ run ---
@pytest.mark.slow
def test_run_study_in_the_background(tmp_path, sample_raw_df: pd.DataFrame) -> None:
    settings = make_project(tmp_path, sample_raw_df)
    prepare_data.run(settings)
    train.run(settings, tracker=NullTracker())
    with TestClient(create_app(settings)) as c:
        response = c.post("/drift/study/run",
                          json={"n_slices": 3, "scenarios": ["natural"], "log_to_mlflow": False})
        assert response.status_code == 202
        assert response.json()["status"] == "running"
        c.app.state.drift_job.wait(STUDY_TIMEOUT_SECONDS)
        status = c.get("/drift/study/status").json()
        assert status["status"] == "succeeded", status
        assert status["params"] == {"n_slices": 3, "scenarios": ["natural"], "log_to_mlflow": False}
        body = c.get("/drift/study").json()
        assert body["n_slices"] == 3 and len(body["scenarios"][0]["steps"]) == 2


def test_run_failure_is_reported(empty_client: TestClient) -> None:
    """No trained model: the job fails and says why instead of crashing the API."""
    assert empty_client.post("/drift/study/run", json={}).status_code == 202
    empty_client.app.state.drift_job.wait(STUDY_TIMEOUT_SECONDS)
    status = empty_client.get("/drift/study/status").json()
    assert status["status"] == "failed" and "make train" in status["error"]


def test_only_one_study_at_a_time(empty_client: TestClient) -> None:
    empty_client.app.state.drift_job.status = "running"
    assert empty_client.post("/drift/study/run", json={}).status_code == 409


@pytest.mark.parametrize("payload", [{"n_slices": 1}, {"n_slices": 50}, {"scenarios": ["nope"]},
                                     {"scenarios": []}, {"unknown": 1}])
def test_invalid_run_requests(empty_client: TestClient, payload: dict) -> None:
    assert empty_client.post("/drift/study/run", json=payload).status_code == 422


def test_mlflow_runs_lists_the_drift_experiment(empty_client: TestClient) -> None:
    body = empty_client.get("/drift/study/mlflow-runs").json()
    assert body["experiment"] == "data-drift" and body["runs"] == []


# -------------------------------------------------------------- compare ---
def test_compare_two_sets_detects_an_age_shift(empty_client: TestClient, raw_df: pd.DataFrame) -> None:
    young, old = raw_df[raw_df["Age"] < 35].head(200), raw_df[raw_df["Age"] > 50].head(200)
    body = empty_client.post("/drift/compare",
                             json={"reference": customers(young), "current": customers(old)}).json()
    assert body["reference_source"] == "request"
    assert body["drift_detected"] and "Age" in body["drifted_features"]
    assert body["features"]["Age"]["level"] == "significant"
    assert body["features"]["Age"]["ks"]["pvalue"] < 0.001
    assert body["features"]["Age"]["stats"]["current"]["mean"] > 50


def test_compare_identical_sets_is_stable(empty_client: TestClient, raw_df: pd.DataFrame) -> None:
    rows = customers(raw_df.head(200))
    body = empty_client.post("/drift/compare", json={"reference": rows, "current": rows}).json()
    assert not body["drift_detected"]
    assert all(f["psi"] == 0 for f in body["features"].values())


@pytest.mark.slow
def test_compare_against_the_model_training_profile(client: TestClient, raw_df: pd.DataFrame) -> None:
    rows = customers(raw_df.sample(300, random_state=3))
    body = client.post("/drift/compare", json={"current": rows}).json()
    assert body["reference_source"] == "model_training_data"
    assert not body["drift_detected"]


def test_compare_without_reference_needs_a_model(empty_client: TestClient, raw_df: pd.DataFrame) -> None:
    response = empty_client.post("/drift/compare", json={"current": customers(raw_df.head(50))})
    assert response.status_code == 503


def test_compare_needs_enough_rows(empty_client: TestClient, raw_df: pd.DataFrame) -> None:
    response = empty_client.post("/drift/compare", json={"current": customers(raw_df.head(5))})
    assert response.status_code == 422


def test_mlflow_store_errors_are_503(empty_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow

    def broken(*args, **kwargs):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(mlflow, "MlflowClient", broken)
    response = empty_client.get("/drift/study/mlflow-runs")
    assert response.status_code == 503
    assert "PermissionError" in response.json()["detail"]
