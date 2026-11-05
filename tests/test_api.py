from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from churnai.api.app import REQUEST_ID_HEADER, create_app
from churnai.config import Settings, load_settings
from churnai.models.bundle import ModelBundle
from tests.conftest import CONFIG_FILE

pytestmark = pytest.mark.slow

TOP_K = 5


@pytest.fixture
def client(trained_project: Settings, bundle: ModelBundle) -> Iterator[TestClient]:
    with TestClient(create_app(trained_project, bundle)) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path) -> Iterator[TestClient]:
    settings = load_settings(config_path=CONFIG_FILE, root=tmp_path)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health_ok(client: TestClient, bundle: ModelBundle) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == bundle.metadata.version


def test_predict_returns_probability_label_and_version(client: TestClient, customer: dict) -> None:
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)
    assert body["churn_prediction"] == (body["churn_probability"] >= body["threshold"])
    assert body["model_version"]


def test_predict_is_deterministic(client: TestClient, customer: dict) -> None:
    first = client.post("/predict", json=customer).json()
    second = client.post("/predict", json=customer).json()
    assert first == second


@pytest.mark.parametrize(
    "override",
    [
        {"age": 5},
        {"geography": "Atlantis"},
        {"num_of_products": 9},
        {"has_cr_card": 2},
        {"unexpected_field": 1},
        {"credit_score": "high"},
    ],
)
def test_invalid_input_returns_422(client: TestClient, customer: dict, override: dict) -> None:
    assert client.post("/predict", json={**customer, **override}).status_code == 422


def test_missing_field_returns_422(client: TestClient, customer: dict) -> None:
    customer.pop("age")
    assert client.post("/predict", json=customer).status_code == 422


def test_batch_predict(client: TestClient, customer: dict) -> None:
    payload = {"customers": [customer, {**customer, "age": 60, "is_active_member": 0}]}
    body = client.post("/predict/batch", json=payload).json()
    assert len(body["predictions"]) == 2


def test_empty_batch_rejected(client: TestClient) -> None:
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_explain_returns_top_features(client: TestClient, customer: dict) -> None:
    body = client.post("/explain", json=customer).json()
    assert len(body["top_features"]) == TOP_K
    assert body["units"] == "log_odds"
    for feature in body["top_features"]:
        assert feature["direction"] in {"increases_churn_risk", "decreases_churn_risk"}


def test_metrics_track_requests_and_drift(client: TestClient, customer: dict) -> None:
    client.post("/predict", json=customer)
    client.post("/predict", json={"age": -1})
    body = client.get("/metrics").json()
    assert body["requests_by_route"]["/predict"] == 2
    assert body["errors_by_route"]["/predict"] == 1
    assert body["latency"]["p95_ms"] is not None
    assert body["drift"]["status"] == "insufficient_data"


def test_request_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "abc123"})
    assert response.headers[REQUEST_ID_HEADER] == "abc123"


def test_model_info(client: TestClient) -> None:
    metadata = client.get("/model").json()["metadata"]
    assert {"model_name", "threshold", "test_metrics", "version"} <= set(metadata)


def test_service_degrades_gracefully_without_model(client_without_model: TestClient, customer: dict) -> None:
    health = client_without_model.get("/health").json()
    assert health["status"] == "unavailable"
    assert client_without_model.post("/predict", json=customer).status_code == 503
