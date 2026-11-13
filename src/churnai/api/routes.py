"""HTTP routes. Deliberately thin: validate, delegate to the service, shape output."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from churnai.api.ab import CUSTOMER_ID_HEADER, ABExperiment
from churnai.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    ExplainResponse,
    FeatureContributionOut,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    to_frame,
)
from churnai.api.service import ModelService, Prediction
from churnai.monitoring.metrics import RequestMetrics

router = APIRouter()


def get_service(request: Request) -> ModelService:
    service: ModelService | None = request.app.state.service
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not loaded")
    return service


@dataclass(frozen=True)
class RoutedRequest:
    """The service that should answer, and the experiment to record the exposure in (if any)."""

    service: ModelService
    customer_id: str | None = None
    experiment: ABExperiment | None = None

    def record(self, prediction: Prediction) -> None:
        if self.experiment and self.customer_id:
            self.experiment.record_prediction(self.customer_id, prediction)


def route_request(
    request: Request,
    customer_id: Annotated[
        str | None, Header(alias=CUSTOMER_ID_HEADER, min_length=1, max_length=64)
    ] = None,
    service: ModelService = Depends(get_service),
) -> RoutedRequest:
    """Requests without a customer ID stay on the champion and outside the experiment."""
    experiment: ABExperiment | None = request.app.state.ab
    if experiment is None or customer_id is None:
        return RoutedRequest(service)
    variant = experiment.assign(customer_id)
    request.state.variant = variant
    return RoutedRequest(experiment.services[variant], customer_id, experiment)


def get_metrics(request: Request) -> RequestMetrics:
    return request.app.state.metrics


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    service: ModelService | None = request.app.state.service
    return HealthResponse(
        status="ok" if service else "unavailable",
        model_loaded=service is not None,
        model_version=service.version if service else None,
        build=request.app.state.build,
        uptime_seconds=round(time.monotonic() - request.app.state.started_at, 1),
    )


@router.get("/metrics", tags=["ops"])
def metrics(
    request: Request, collector: RequestMetrics = Depends(get_metrics)
) -> dict[str, Any]:
    service: ModelService | None = request.app.state.service
    return {
        **collector.snapshot(),
        "model_version": service.version if service else None,
        "drift": asdict(service.drift.report()) if service else None,
        "ab_test": request.app.state.ab.snapshot() if request.app.state.ab else None,
    }


@router.get("/model", response_model=ModelInfoResponse, tags=["model"])
def model_info(service: ModelService = Depends(get_service)) -> ModelInfoResponse:
    return ModelInfoResponse(metadata=service.bundle.metadata.to_dict())


@router.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict(
    customer: CustomerFeatures,  # type: ignore[valid-type]
    routed: RoutedRequest = Depends(route_request),
) -> PredictionResponse:
    (prediction,) = routed.service.predict(to_frame([customer]))
    routed.record(prediction)
    return PredictionResponse(**asdict(prediction))


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["model"])
def predict_batch(
    payload: BatchPredictionRequest, service: ModelService = Depends(get_service)
) -> BatchPredictionResponse:
    predictions = service.predict(to_frame(payload.customers))
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**asdict(p)) for p in predictions]
    )


@router.post(
    "/feedback", response_model=FeedbackResponse, status_code=status.HTTP_202_ACCEPTED, tags=["ab"]
)
def feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
    """Report whether a customer actually churned; joined to their prediction by the analysis."""
    experiment: ABExperiment | None = request.app.state.ab
    if experiment is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No A/B test is running")
    experiment.record_feedback(payload.customer_id, payload.exited)
    return FeedbackResponse(status="recorded", customer_id=payload.customer_id)


@router.post("/explain", response_model=ExplainResponse, tags=["model"])
def explain(
    customer: CustomerFeatures,  # type: ignore[valid-type]
    routed: RoutedRequest = Depends(route_request),
) -> ExplainResponse:
    ((prediction, explanation),) = routed.service.explain(to_frame([customer]))
    routed.record(prediction)
    return ExplainResponse(
        **asdict(prediction),
        base_value=explanation.base_value,
        units=explanation.units,
        top_features=[
            FeatureContributionOut(
                feature=c.feature, value=c.value, shap_value=c.shap_value, direction=c.direction
            )
            for c in explanation.contributions
        ],
    )
