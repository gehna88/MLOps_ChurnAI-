"""Build the shared model pipeline: features -> preprocessing -> estimator."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnai.data.schema import BINARY_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from churnai.features.engineering import (
    ENGINEERED_BINARY,
    ENGINEERED_NUMERIC,
    ChurnFeatureEngineer,
)

FEATURES_STEP = "features"
PREPROCESS_STEP = "preprocess"
MODEL_STEP = "model"

NUMERIC_BLOCK = "num"
BINARY_BLOCK = "bin"
CATEGORICAL_BLOCK = "cat"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (NUMERIC_BLOCK, StandardScaler(), [*NUMERIC_COLUMNS, *ENGINEERED_NUMERIC]),
            (BINARY_BLOCK, "passthrough", [*BINARY_COLUMNS, *ENGINEERED_BINARY]),
            (
                CATEGORICAL_BLOCK,
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_COLUMNS),
            ),
        ],
        remainder="drop",
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Every candidate model plugs into the same preprocessing pipeline."""
    return Pipeline(
        steps=[
            (FEATURES_STEP, ChurnFeatureEngineer()),
            (PREPROCESS_STEP, build_preprocessor()),
            (MODEL_STEP, estimator),
        ]
    )


def source_features(preprocessor: ColumnTransformer) -> list[str]:
    """Map each transformed output column back to the feature it came from.

    One-hot columns such as ``Geography=Germany`` map to ``Geography`` so that
    explanations can be aggregated to human-meaningful features.
    """
    sources: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if name == CATEGORICAL_BLOCK:
            for column, categories in zip(columns, transformer.categories_, strict=True):
                sources.extend([column] * len(categories))
        else:
            sources.extend(columns)
    return sources
