"""Typed, validated configuration loaded from YAML.

Configuration is resolved relative to a project root so that no absolute paths
are ever hardcoded:

* ``CHURNAI_HOME``   - project root (defaults to the current working directory)
* ``CHURNAI_CONFIG`` - config file (defaults to ``<root>/configs/config.yaml``)

Selected values can also be overridden per environment (see ``ENV_OVERRIDES``),
e.g. pointing ``CHURNAI_MLFLOW_TRACKING_URI`` at a remote MLflow server.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from churnai.exceptions import ConfigError

ENV_HOME = "CHURNAI_HOME"
ENV_CONFIG = "CHURNAI_CONFIG"
DEFAULT_CONFIG_RELPATH = Path("configs") / "config.yaml"

# Environment variable -> (config section, key) overrides.
ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "CHURNAI_MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri"),
    "CHURNAI_MLFLOW_ARTIFACT_DIR": ("mlflow", "artifact_dir"),
    "CHURNAI_MODEL_DIR": ("paths", "model_dir"),
    "CHURNAI_AB_ENABLED": ("ab_test", "enabled"),
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProjectConfig(_Frozen):
    name: str
    random_seed: int


class PathsConfig(_Frozen):
    raw_data: Path
    processed_dir: Path
    rejected_dir: Path
    reports_dir: Path
    model_dir: Path


class DataConfig(_Frozen):
    test_size: float = Field(gt=0, lt=1)
    identifier_columns: list[str]
    leakage_columns: list[str]
    max_rejected_fraction: float = Field(ge=0, le=1)


class TrainingConfig(_Frozen):
    models_config: Path
    tuning_config: Path
    cv_folds: int = Field(ge=2)
    primary_metric: str
    leakage_auc_threshold: float = Field(gt=0.5, le=1.0)
    threshold_metric: str


class MlflowConfig(_Frozen):
    tracking_uri: str
    artifact_dir: Path
    pip_requirements: Path
    experiment_name: str
    tuning_experiment: str
    ab_experiment: str
    registered_model_name: str
    champion_alias: str
    challenger_alias: str


class ExplainConfig(_Frozen):
    top_k: int = Field(ge=1)
    background_size: int = Field(ge=10)
    report_sample_size: int = Field(ge=10)


class MonitoringConfig(_Frozen):
    psi_bins: int = Field(ge=2)
    psi_alert_threshold: float = Field(gt=0)
    drift_window_size: int = Field(ge=1)
    drift_min_samples: int = Field(ge=1)
    latency_window_size: int = Field(ge=1)


class ABTestConfig(_Frozen):
    enabled: bool
    split_pct: int = Field(ge=0, le=100)
    bundle_dir: Path
    predictions_log: Path
    feedback_log: Path
    cv_repeats: int = Field(ge=2)
    bootstrap_iterations: int = Field(ge=100)
    alpha: float = Field(gt=0, lt=1)
    power: float = Field(gt=0, lt=1)
    min_detectable_recall_diff: float = Field(gt=0, lt=1)
    cost_false_negative: float = Field(ge=0)
    cost_false_positive: float = Field(ge=0)


class DriftYear(_Frozen):
    """One simulated year. Neutral values (0, 1, 1, 0) reproduce the test split unchanged."""

    name: str
    age_tilt: float = Field(ge=0)  # log-weight per standard deviation of age
    germany_weight: float = Field(gt=0)
    inactive_weight: float = Field(gt=0)
    salary_growth: float = Field(ge=0)  # e.g. 0.09 = salaries 9% higher

    @property
    def is_neutral(self) -> bool:
        return (self.age_tilt, self.germany_weight, self.inactive_weight, self.salary_growth) == (
            0, 1, 1, 0
        )


class DriftSimulationConfig(_Frozen):
    batch_dir: Path
    batch_size: int = Field(ge=1)
    years: list[DriftYear] = Field(min_length=1)


class Settings(_Frozen):
    """Root configuration object. Use :func:`get_settings` to obtain one."""

    root: Path
    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    training: TrainingConfig
    mlflow: MlflowConfig
    explain: ExplainConfig
    monitoring: MonitoringConfig
    ab_test: ABTestConfig
    drift_simulation: DriftSimulationConfig

    def resolve(self, relative: Path | str) -> Path:
        """Resolve a project-relative path against the project root."""
        path = Path(relative)
        return path if path.is_absolute() else (self.root / path).resolve()

    # Convenience accessors for frequently used artifact locations.
    @property
    def model_dir(self) -> Path:
        return self.resolve(self.paths.model_dir)

    @property
    def model_path(self) -> Path:
        """The bundle the API serves: always the latest champion."""
        return self.model_dir / "model.joblib"

    def versioned_model_path(self, version: str) -> Path:
        """Immutable, versioned copy of a champion, kept for audit and rollback."""
        return self.model_dir / f"{version}.joblib"

    def ab_bundle_path(self, variant: str) -> Path:
        """Local copy of the bundle behind an A/B variant, fetched from its MLflow alias."""
        return self.resolve(self.ab_test.bundle_dir) / f"{variant}.joblib"

    @property
    def processed_dir(self) -> Path:
        return self.resolve(self.paths.processed_dir)

    @property
    def train_path(self) -> Path:
        return self.processed_dir / "train.csv"

    @property
    def test_path(self) -> Path:
        return self.processed_dir / "test.csv"

    @property
    def mlflow_tracking_uri(self) -> str:
        """SQLite URIs are made absolute so MLflow works from any cwd."""
        uri = self.mlflow.tracking_uri
        prefix = "sqlite:///"
        if uri.startswith(prefix) and not Path(uri[len(prefix) :]).is_absolute():
            return f"{prefix}{self.resolve(uri[len(prefix):]).as_posix()}"
        return uri

    @property
    def mlflow_artifact_uri(self) -> str:
        return self.resolve(self.mlflow.artifact_dir).as_uri()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    if not isinstance(content, dict):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    return content


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Let deployment environments override selected values without editing YAML."""
    for env_var, (section, key) in ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            raw.setdefault(section, {})[key] = value
    return raw


def load_settings(config_path: Path | None = None, root: Path | None = None) -> Settings:
    """Load and validate settings without caching (useful in tests)."""
    root = Path(root or os.environ.get(ENV_HOME) or Path.cwd()).resolve()
    config_path = Path(config_path or os.environ.get(ENV_CONFIG) or root / DEFAULT_CONFIG_RELPATH)
    try:
        return Settings(root=root, **_apply_env_overrides(_read_yaml(config_path)))
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {config_path}:\n{exc}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings resolved from the environment."""
    return load_settings()


def load_yaml(path: Path) -> dict[str, Any]:
    """Public helper for secondary YAML files such as the model catalogue."""
    return _read_yaml(path)
