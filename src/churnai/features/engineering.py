"""Domain feature engineering as a stateless scikit-learn transformer.

Living inside the Pipeline means the API computes exactly the same features as
training did - there is no separate serving-side implementation to drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Guards divisions against zero salaries/ages without distorting real values.
EPSILON = 1.0

BALANCE_IS_ZERO = "BalanceIsZero"
BALANCE_SALARY_RATIO = "BalanceSalaryRatio"
TENURE_AGE_RATIO = "TenureAgeRatio"

ENGINEERED_NUMERIC: tuple[str, ...] = (BALANCE_SALARY_RATIO, TENURE_AGE_RATIO)
ENGINEERED_BINARY: tuple[str, ...] = (BALANCE_IS_ZERO,)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add engineered features to a frame of contract features.

    * ``BalanceIsZero``      - over a third of customers hold no balance, a
                               distinct behavioural segment.
    * ``BalanceSalaryRatio`` - balance relative to income (financial engagement).
    * ``TenureAgeRatio``     - share of adult life spent as a customer (loyalty).
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ChurnFeatureEngineer:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out[BALANCE_IS_ZERO] = (out["Balance"] == 0).astype(int)
        out[BALANCE_SALARY_RATIO] = out["Balance"] / (out["EstimatedSalary"] + EPSILON)
        out[TENURE_AGE_RATIO] = out["Tenure"] / (out["Age"] + EPSILON)
        return out

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        base = list(input_features if input_features is not None else self.feature_names_in_)
        return np.asarray([*base, *ENGINEERED_BINARY, *ENGINEERED_NUMERIC], dtype=object)
