"""Request/response models.

``CustomerFeatures`` is generated from the data contract
(:data:`churnai.data.schema.FEATURES`), so API validation uses exactly the same
types, ranges and categories as offline data validation.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model

from churnai.data.schema import API_TO_COLUMN, FEATURES, FeatureSpec

MAX_BATCH_SIZE = 1000

EXAMPLE_CUSTOMER: dict[str, Any] = {
    "credit_score": 619,
    "geography": "France",
    "gender": "Female",
    "age": 42,
    "tenure": 2,
    "balance": 0.0,
    "num_of_products": 1,
    "has_cr_card": 1,
    "is_active_member": 1,
    "estimated_salary": 101348.88,
    "satisfaction_score": 2,
    "card_type": "DIAMOND",
    "point_earned": 464,
}


def _field_for(spec: FeatureSpec) -> tuple[Any, Any]:
    if spec.is_categorical:
        return Literal[spec.categories], Field(description=spec.description)  # type: ignore[valid-type]
    python_type = int if spec.integer else float
    return python_type, Field(ge=spec.min_value, le=spec.max_value, description=spec.description)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_CUSTOMER})


CustomerFeatures: type[BaseModel] = create_model(  # type: ignore[call-overload]
    "CustomerFeatures",
    __base__=_StrictModel,
    **{spec.api_name: _field_for(spec) for spec in FEATURES},
)


def to_frame(customers: list[BaseModel]) -> pd.DataFrame:
    """Convert validated API payloads into a frame with training column names."""
    return pd.DataFrame([c.model_dump() for c in customers]).rename(columns=API_TO_COLUMN)


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=MAX_BATCH_SIZE)  # type: ignore[valid-type]


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn_prediction: bool
    threshold: float
    model_version: str
    model_variant: str = "A"


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class FeedbackRequest(BaseModel):
    """The observed outcome for a customer, reported once it is known."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, max_length=64)
    exited: Literal[0, 1]


class FeedbackResponse(BaseModel):
    status: Literal["recorded"]
    customer_id: str


class FeatureContributionOut(BaseModel):
    feature: str
    value: Any
    shap_value: float
    direction: str


class ExplainResponse(PredictionResponse):
    base_value: float
    units: str
    top_features: list[FeatureContributionOut]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None
    build: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    metadata: dict[str, Any]
