"""The API with an A/B test switched on: routing, logging, feedback and per-variant stats."""

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from churnai.api.ab import CUSTOMER_ID_HEADER, VARIANT_A, VARIANT_B, bucket
from churnai.api.app import create_app
from churnai.config import Settings
from churnai.models.bundle import ModelBundle

pytestmark = pytest.mark.slow


def ab_settings(settings: Settings, tmp_path: Path, enabled: bool = True, split_pct: int = 50) -> Settings:
    ab = settings.ab_test.model_copy(update={
        "enabled": enabled,
        "split_pct": split_pct,
        "bundle_dir": tmp_path / "ab",
        "predictions_log": tmp_path / "predictions.jsonl",
        "feedback_log": tmp_path / "feedback.jsonl",
    })
    return settings.model_copy(update={"ab_test": ab})


@pytest.fixture
def challenger(bundle: ModelBundle) -> ModelBundle:
    """Same pipeline under another name, so a response shows which variant answered."""
    return replace(bundle, metadata=replace(bundle.metadata, model_name="challenger"))


@pytest.fixture
def client(
    trained_project: Settings, bundle: ModelBundle, challenger: ModelBundle, tmp_path: Path
) -> Iterator[TestClient]:
    settings = ab_settings(trained_project, tmp_path)
    with TestClient(create_app(settings, bundle, challenger)) as test_client:
        yield test_client


def customer_in(variant: str) -> str:
    """A customer ID that the 50/50 split routes to ``variant``."""
    for i in range(1000):
        if (bucket(str(i)) < 50) == (variant == VARIANT_B):
            return str(i)
    raise AssertionError("no customer found")


def predict(client: TestClient, customer: dict, customer_id: str | None) -> dict:
    headers = {CUSTOMER_ID_HEADER: customer_id} if customer_id else {}
    response = client.post("/predict", json=customer, headers=headers)
    assert response.status_code == 200
    return response.json()


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("variant", [VARIANT_A, VARIANT_B])
def test_customer_is_served_by_their_variant_every_time(client, customer, variant) -> None:
    customer_id = customer_in(variant)
    responses = [predict(client, customer, customer_id) for _ in range(3)]
    assert {r["model_variant"] for r in responses} == {variant}
    assert responses[0]["model_version"].startswith("challenger") == (variant == VARIANT_B)


def test_split_pct_controls_traffic_share(trained_project, bundle, challenger, tmp_path, customer) -> None:
    settings = ab_settings(trained_project, tmp_path, split_pct=0)
    with TestClient(create_app(settings, bundle, challenger)) as client:
        assert predict(client, customer, customer_in(VARIANT_B))["model_variant"] == VARIANT_A
    settings = ab_settings(trained_project, tmp_path, split_pct=100)
    with TestClient(create_app(settings, bundle, challenger)) as client:
        assert predict(client, customer, customer_in(VARIANT_A))["model_variant"] == VARIANT_B


def test_request_without_customer_id_stays_out_of_the_experiment(client, customer, tmp_path) -> None:
    assert predict(client, customer, None)["model_variant"] == VARIANT_A
    assert not (tmp_path / "predictions.jsonl").exists()


def test_prediction_and_feedback_are_logged_for_joining(client, customer, tmp_path) -> None:
    customer_id = customer_in(VARIANT_B)
    body = predict(client, customer, customer_id)
    response = client.post("/feedback", json={"customer_id": customer_id, "exited": 1})
    assert response.status_code == 202
    (logged,) = read_log(tmp_path / "predictions.jsonl")
    assert logged["customer_id"] == customer_id
    assert logged["model_variant"] == VARIANT_B
    assert logged["churn_probability"] == body["churn_probability"]
    assert read_log(tmp_path / "feedback.jsonl") == [{"customer_id": customer_id, "exited": 1}]


@pytest.mark.parametrize(
    "payload", [{"customer_id": "1", "exited": 2}, {"customer_id": "", "exited": 1}, {"exited": 1}]
)
def test_feedback_rejects_invalid_payloads(client, payload) -> None:
    assert client.post("/feedback", json=payload).status_code == 422


def test_explain_uses_the_same_variant_as_predict(client, customer) -> None:
    headers = {CUSTOMER_ID_HEADER: customer_in(VARIANT_B)}
    assert client.post("/explain", json=customer, headers=headers).json()["model_variant"] == VARIANT_B


def test_metrics_report_latency_and_drift_per_variant(client, customer) -> None:
    predict(client, customer, customer_in(VARIANT_A))
    predict(client, customer, customer_in(VARIANT_B))
    predict(client, customer, customer_in(VARIANT_B))
    section = client.get("/metrics").json()["ab_test"]
    assert section["split_pct"] == 50
    a, b = section["variants"][VARIANT_A], section["variants"][VARIANT_B]
    assert a["requests"]["requests_total"] == 1
    assert b["requests"]["requests_total"] == 2
    assert b["requests"]["latency"]["avg_ms"] > 0
    assert b["model_version"].startswith("challenger")
    assert b["drift"]["window_size"] == 2
    assert a["drift"]["window_size"] == 1


def test_disabled_ab_test_serves_champion_alone(trained_project, bundle, tmp_path, customer) -> None:
    settings = ab_settings(trained_project, tmp_path, enabled=False)
    with TestClient(create_app(settings, bundle)) as client:
        assert predict(client, customer, customer_in(VARIANT_B))["model_variant"] == VARIANT_A
        assert client.post("/feedback", json={"customer_id": "1", "exited": 1}).status_code == 404
        assert client.get("/metrics").json()["ab_test"] is None
    assert not (tmp_path / "predictions.jsonl").exists()


def test_missing_challenger_falls_back_to_champion(trained_project, bundle, tmp_path, customer) -> None:
    settings = ab_settings(trained_project, tmp_path)
    with TestClient(create_app(settings, bundle)) as client:
        assert predict(client, customer, customer_in(VARIANT_B))["model_variant"] == VARIANT_A
        assert client.get("/metrics").json()["ab_test"] is None
