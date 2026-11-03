"""Reading and writing tabular data and small JSON artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from churnai.data.schema import TARGET
from churnai.exceptions import DataValidationError


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataValidationError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise DataValidationError(f"Data file is empty: {path}")
    return df


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_features_target(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise DataValidationError(f"Target column '{target}' not found")
    return df.drop(columns=[target]), df[target].astype(int)
