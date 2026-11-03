"""Shared fixtures.

Each test session builds an isolated project root in a temp directory with a
sample of the real dataset and a fast model catalogue, so tests never touch
real artefacts and do not depend on `make train` having been run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from churnai.api.schemas import EXAMPLE_CUSTOMER
from churnai.config import Settings, load_settings
from churnai.data.schema import TARGET
from churnai.models.bundle import ModelBundle
from churnai.models.tracking import NullTracker
from churnai.pipelines import prepare_data, train

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = REPO_ROOT / "data" / "raw" / "Customer-Churn-Records.csv"
CONFIG_FILE = REPO_ROOT / "configs" / "config.yaml"
SAMPLE_ROWS = 2500
SEED = 7

FAST_MODELS = {
    "candidates": [
        {"name": "logreg", "estimator": "logistic_regression",
         "params": {"max_iter": 1000, "class_weight": "balanced"}},
        {"name": "xgb_small", "estimator": "xgboost",
         "params": {"n_estimators": 40, "max_depth": 3, "scale_pos_weight": "auto"}},
        {"name": "lgbm_small", "estimator": "lightgbm",
         "params": {"n_estimators": 40, "num_leaves": 15, "class_weight": "balanced"}},
    ]
}


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA)


@pytest.fixture(scope="session")
def sample_raw_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Stratified sample keeps the churn rate realistic while staying fast."""
    return (
        raw_df.groupby(TARGET, group_keys=False)
        .apply(lambda g: g.sample(frac=SAMPLE_ROWS / len(raw_df), random_state=SEED))
        .reset_index(drop=True)
    )


def make_project(root: Path, raw: pd.DataFrame, models: dict | None = None) -> Settings:
    """Lay out a minimal project in ``root`` and return settings bound to it."""
    (root / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy(CONFIG_FILE, root / "configs" / "config.yaml")
    shutil.copytree(REPO_ROOT / "requirements", root / "requirements", dirs_exist_ok=True)
    (root / "configs" / "models.yaml").write_text(yaml.safe_dump(models or FAST_MODELS))
    raw_path = root / "data" / "raw" / RAW_DATA.name
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    return load_settings(root=root)


@pytest.fixture
def project(tmp_path: Path, sample_raw_df: pd.DataFrame) -> Settings:
    return make_project(tmp_path, sample_raw_df)


@pytest.fixture(scope="session")
def trained_project(tmp_path_factory: pytest.TempPathFactory, sample_raw_df: pd.DataFrame) -> Settings:
    """A project that has been prepared and trained once per session."""
    settings = make_project(tmp_path_factory.mktemp("trained"), sample_raw_df)
    prepare_data.run(settings)
    train.run(settings, tracker=NullTracker())
    return settings


@pytest.fixture(scope="session")
def bundle(trained_project: Settings) -> ModelBundle:
    return ModelBundle.load(trained_project.model_path)


@pytest.fixture
def customer() -> dict:
    return dict(EXAMPLE_CUSTOMER)
