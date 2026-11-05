"""Row-level data validation driven by the data contract.

Validation is split into two levels:

* **Schema** - required columns must exist. A missing column means the upstream
  export is broken, so this raises immediately.
* **Rows** - each row is checked for nulls, types, ranges, categories, target
  values and duplicate identifiers. Failing rows are *rejected* (kept aside with
  a reason) rather than silently dropped, so they can be audited.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from churnai.data.schema import FEATURES, TARGET, TARGET_VALUES, FeatureSpec
from churnai.exceptions import DataValidationError
from churnai.logging_utils import get_logger

logger = get_logger(__name__)

REJECTION_REASON_COLUMN = "rejection_reason"
REASON_SEPARATOR = "; "

# A predicate returns a boolean Series that is True where a row FAILS the check.
FailurePredicate = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class RowCheck:
    name: str
    fails: FailurePredicate


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    rejected: pd.DataFrame
    failure_counts: dict[str, int] = field(default_factory=dict)

    @property
    def rejected_fraction(self) -> float:
        total = len(self.valid) + len(self.rejected)
        return len(self.rejected) / total if total else 0.0


# ------------------------------------------------------------ check builders --


def not_null(column: str) -> RowCheck:
    return RowCheck(f"{column}:null", lambda df: df[column].isna())


def in_range(spec: FeatureSpec) -> RowCheck:
    def fails(df: pd.DataFrame) -> pd.Series:
        values = df[spec.column]
        too_low = values < spec.min_value if spec.min_value is not None else False
        too_high = values > spec.max_value if spec.max_value is not None else False
        return values.notna() & (too_low | too_high)

    return RowCheck(f"{spec.column}:out_of_range", fails)


def is_integer(spec: FeatureSpec) -> RowCheck:
    def fails(df: pd.DataFrame) -> pd.Series:
        values = df[spec.column]
        return values.notna() & (np.floor(values) != values)

    return RowCheck(f"{spec.column}:not_integer", fails)


def in_categories(spec: FeatureSpec) -> RowCheck:
    allowed = set(spec.categories)
    return RowCheck(
        f"{spec.column}:unknown_category",
        lambda df: df[spec.column].notna() & ~df[spec.column].isin(allowed),
    )


def target_is_valid(target: str = TARGET, allowed: Sequence[int] = TARGET_VALUES) -> RowCheck:
    return RowCheck(f"{target}:invalid", lambda df: ~df[target].isin(list(allowed)))


def unique(column: str) -> RowCheck:
    """Reject every occurrence of a duplicated key after the first."""
    return RowCheck(f"{column}:duplicate", lambda df: df[column].duplicated(keep="first"))


def checks_for_feature(spec: FeatureSpec) -> list[RowCheck]:
    checks = [not_null(spec.column)]
    if spec.is_categorical:
        checks.append(in_categories(spec))
    else:
        checks.append(in_range(spec))
        if spec.integer:
            checks.append(is_integer(spec))
    return checks


# ----------------------------------------------------------------- validator --


class DataValidator:
    """Apply schema and row checks to a DataFrame.

    Parameters
    ----------
    features:
        Feature specifications to validate against.
    require_target:
        Whether the target column must be present (True for training data,
        False for inference payloads).
    unique_key:
        Optional identifier column that must be unique.
    """

    def __init__(
        self,
        features: Iterable[FeatureSpec] = FEATURES,
        require_target: bool = True,
        unique_key: str | None = None,
    ) -> None:
        self.features = tuple(features)
        self.require_target = require_target
        self.unique_key = unique_key
        self.checks = self._build_checks()

    @property
    def required_columns(self) -> list[str]:
        columns = [spec.column for spec in self.features]
        if self.require_target:
            columns.append(TARGET)
        return columns

    def _build_checks(self) -> list[RowCheck]:
        checks = [check for spec in self.features for check in checks_for_feature(spec)]
        if self.require_target:
            checks.append(target_is_valid())
        if self.unique_key:
            checks.append(unique(self.unique_key))
        return checks

    def validate_schema(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.required_columns if col not in df.columns]
        if self.unique_key and self.unique_key not in df.columns:
            missing.append(self.unique_key)
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")

    def _coerce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Non-numeric junk in numeric columns becomes NaN and fails `not_null`."""
        df = df.copy()
        for spec in self.features:
            if not spec.is_categorical:
                df[spec.column] = pd.to_numeric(df[spec.column], errors="coerce")
        return df

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        self.validate_schema(df)
        df = self._coerce_numeric(df)

        reasons = pd.Series("", index=df.index, dtype="object")
        failure_counts: dict[str, int] = {}
        for check in self.checks:
            failed = check.fails(df).fillna(False).astype(bool)
            count = int(failed.sum())
            if count:
                failure_counts[check.name] = count
                reasons[failed] = reasons[failed] + check.name + REASON_SEPARATOR

        rejected_mask = reasons != ""
        rejected = df[rejected_mask].copy()
        rejected[REJECTION_REASON_COLUMN] = reasons[rejected_mask].str.rstrip(REASON_SEPARATOR)
        valid = df[~rejected_mask].copy()

        if failure_counts:
            logger.warning(
                "Rows rejected during validation",
                extra={"fields": {"rejected": len(rejected), "failures": failure_counts}},
            )
        return ValidationResult(valid=valid, rejected=rejected, failure_counts=failure_counts)
