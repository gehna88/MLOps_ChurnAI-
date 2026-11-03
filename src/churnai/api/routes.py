"""HTTP routes. Deliberately thin: validate, delegate to the service, shape output."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from churnai.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    ExplainResponse,
    FeatureContributionOut,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    to_frame,
)
from churnai.api.service import ModelService
from churnai.monitoring.metrics import RequestMetrics

router = APIRouter()


def get_service(request: Request) -> ModelService:
    service: ModelService | None = request.app.state.service
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not loaded")
    return service


def get_metrics(request: Request) -> RequestMetrics:
    return request.app.state.metrics


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    service: ModelService | None = request.app.state.service
    return HealthResponse(
        status="ok" if service else "unavailable",
        model_loaded=service is not None,
        model_version=service.version if service else None,
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
    }


@router.get("/model", response_model=ModelInfoResponse, tags=["model"])
def model_info(service: ModelService = Depends(get_service)) -> ModelInfoResponse:
    return ModelInfoResponse(metadata=service.bundle.metadata.to_dict())


@router.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict(
    customer: CustomerFeatures,  # type: ignore[valid-type]
    service: ModelService = Depends(get_service),
) -> PredictionResponse:
    (prediction,) = service.predict(to_frame([customer]))
    return PredictionResponse(**asdict(prediction))


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["model"])
def predict_batch(
    payload: BatchPredictionRequest, service: ModelService = Depends(get_service)
) -> BatchPredictionResponse:
    predictions = service.predict(to_frame(payload.customers))
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**asdict(p)) for p in predictions]
    )


@router.post("/explain", response_model=ExplainResponse, tags=["model"])
def explain(
    customer: CustomerFeatures,  # type: ignore[valid-type]
    service: ModelService = Depends(get_service),
) -> ExplainResponse:
    ((prediction, explanation),) = service.explain(to_frame([customer]))
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
