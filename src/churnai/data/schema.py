"""The data contract: the single source of truth for model features.

Both the offline validator and the online API request schema are generated
from :data:`FEATURES`, so training-time and serving-time checks cannot drift.
Bounds are *domain-plausible* limits, deliberately wider than the values seen in
the training sample, so that legitimate new customers are not rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FeatureKind(StrEnum):
    NUMERIC = "numeric"
    BINARY = "binary"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class FeatureSpec:
    """Description and constraints of a single model input."""

    column: str  # column name in the raw CSV
    api_name: str  # snake_case field name exposed by the API
    kind: FeatureKind
    description: str
    min_value: float | None = None
    max_value: float | None = None
    categories: tuple[str, ...] = ()
    integer: bool = False

    @property
    def is_categorical(self) -> bool:
        return self.kind is FeatureKind.CATEGORICAL


TARGET = "Exited"
TARGET_VALUES = (0, 1)

# Domain bounds, named so they are not magic numbers scattered around the code.
CREDIT_SCORE_RANGE = (300, 900)
AGE_RANGE = (18, 100)
TENURE_RANGE = (0, 60)
MONEY_MAX = 10_000_000.0
PRODUCTS_RANGE = (1, 4)
SATISFACTION_RANGE = (1, 5)
POINTS_RANGE = (0, 10_000)
BINARY_RANGE = (0, 1)

GEOGRAPHIES = ("France", "Germany", "Spain")
GENDERS = ("Female", "Male")
CARD_TYPES = ("SILVER", "GOLD", "PLATINUM", "DIAMOND")

FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("CreditScore", "credit_score", FeatureKind.NUMERIC,
                "Customer credit score", *CREDIT_SCORE_RANGE, integer=True),
    FeatureSpec("Geography", "geography", FeatureKind.CATEGORICAL,
                "Country of residence", categories=GEOGRAPHIES),
    FeatureSpec("Gender", "gender", FeatureKind.CATEGORICAL,
                "Customer gender", categories=GENDERS),
    FeatureSpec("Age", "age", FeatureKind.NUMERIC,
                "Age in years", *AGE_RANGE, integer=True),
    FeatureSpec("Tenure", "tenure", FeatureKind.NUMERIC,
                "Years as a client of the bank", *TENURE_RANGE, integer=True),
    FeatureSpec("Balance", "balance", FeatureKind.NUMERIC,
                "Account balance", 0.0, MONEY_MAX),
    FeatureSpec("NumOfProducts", "num_of_products", FeatureKind.NUMERIC,
                "Number of bank products held", *PRODUCTS_RANGE, integer=True),
    FeatureSpec("HasCrCard", "has_cr_card", FeatureKind.BINARY,
                "1 if the customer holds a credit card", *BINARY_RANGE, integer=True),
    FeatureSpec("IsActiveMember", "is_active_member", FeatureKind.BINARY,
                "1 if the customer is an active member", *BINARY_RANGE, integer=True),
    FeatureSpec("EstimatedSalary", "estimated_salary", FeatureKind.NUMERIC,
                "Estimated annual salary", 0.0, MONEY_MAX),
    FeatureSpec("Satisfaction Score", "satisfaction_score", FeatureKind.NUMERIC,
                "Complaint-resolution satisfaction (1-5)", *SATISFACTION_RANGE, integer=True),
    FeatureSpec("Card Type", "card_type", FeatureKind.CATEGORICAL,
                "Card tier", categories=CARD_TYPES),
    FeatureSpec("Point Earned", "point_earned", FeatureKind.NUMERIC,
                "Loyalty points earned with the card", *POINTS_RANGE, integer=True),
)

FEATURE_COLUMNS: tuple[str, ...] = tuple(f.column for f in FEATURES)
NUMERIC_COLUMNS: tuple[str, ...] = tuple(f.column for f in FEATURES if f.kind is FeatureKind.NUMERIC)
BINARY_COLUMNS: tuple[str, ...] = tuple(f.column for f in FEATURES if f.kind is FeatureKind.BINARY)
CATEGORICAL_COLUMNS: tuple[str, ...] = tuple(f.column for f in FEATURES if f.is_categorical)
API_TO_COLUMN: dict[str, str] = {f.api_name: f.column for f in FEATURES}


def get_feature(column: str) -> FeatureSpec:
    for spec in FEATURES:
        if spec.column == column:
            return spec
    raise KeyError(f"Unknown feature column: {column}")
