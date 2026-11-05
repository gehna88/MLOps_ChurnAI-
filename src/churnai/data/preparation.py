"""Pure transformations that turn a validated raw frame into a modelling frame."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from churnai.data.schema import FEATURE_COLUMNS, TARGET
from churnai.exceptions import DataValidationError


def drop_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Drop columns if present; absent columns are ignored (idempotent)."""
    return df.drop(columns=[c for c in columns if c in df.columns])


def select_model_columns(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Keep only contract features (+ target), in a deterministic order.

    Selecting an allow-list rather than dropping a deny-list guarantees that any
    new upstream column - including a future leaky one - never reaches training.
    """
    columns = [*FEATURE_COLUMNS, TARGET] if include_target else list(FEATURE_COLUMNS)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"Cannot select model columns, missing: {missing}")
    return df[columns].copy()


def assert_no_leakage(df: pd.DataFrame, forbidden: Iterable[str]) -> None:
    leaked = sorted(set(forbidden) & set(df.columns))
    if leaked:
        raise DataValidationError(f"Forbidden columns present in modelling data: {leaked}")


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


def stratified_split(df: pd.DataFrame, test_size: float, seed: int, target: str = TARGET) -> Split:
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[target]
    )
    return Split(train=train.reset_index(drop=True), test=test.reset_index(drop=True))
