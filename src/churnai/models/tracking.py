"""Experiment tracking behind a small interface.

Training code depends on :class:`ExperimentTracker`, not on MLflow directly, so
it can be unit-tested with :class:`NullTracker` and MLflow stays out of the
serving image. MLflow is imported lazily inside :class:`MlflowTracker`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from sklearn.pipeline import Pipeline

from churnai.logging_utils import get_logger, preserve_logging
from churnai.models.bundle import POSITIVE_CLASS_INDEX

logger = get_logger(__name__)


class ExperimentTracker(Protocol):
    @contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None) -> Iterator[str | None]: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None: ...
    def log_dict(self, payload: dict[str, Any], filename: str) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def log_model(self, pipeline: Pipeline, example: Any) -> None: ...
    def register(self, run_id: str) -> str | None: ...


class NullTracker:
    """No-op tracker for tests and dry runs."""

    @contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None) -> Iterator[str | None]:
        yield None

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        pass

    def log_dict(self, payload: dict[str, Any], filename: str) -> None:
        pass

    def log_artifact(self, path: Path) -> None:
        pass

    def log_model(self, pipeline: Pipeline, example: Any) -> None:
        pass

    def register(self, run_id: str) -> str | None:
        return None


def applicable_requirements(path: Path) -> list[str]:
    """Read a requirements file, keeping only lines whose markers match this platform.

    MLflow does not evaluate environment markers when it checks the current
    environment, so e.g. a Linux-only ``xgboost-cpu`` line would raise spurious
    mismatch warnings on Windows.
    """
    from packaging.requirements import Requirement

    requirements = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate():
            requirement.marker = None
            requirements.append(str(requirement))
    return requirements


class MlflowTracker:
    """MLflow-backed tracker with model registry support."""

    MODEL_ARTIFACT_PATH = "model"

    def __init__(
        self,
        tracking_uri: str,
        artifact_uri: str,
        experiment: str,
        registered_model: str,
        champion_alias: str,
        pip_requirements: Path | None = None,
    ) -> None:
        import mlflow  # lazy: keeps MLflow out of the serving runtime

        self._mlflow = mlflow
        self.registered_model = registered_model
        self.champion_alias = champion_alias
        # Explicit requirements stop MLflow guessing (and warning about our local package).
        self.pip_requirements = applicable_requirements(pip_requirements) if pip_requirements else None
        # A fresh MLflow database runs migrations that reconfigure logging; undo that.
        with preserve_logging():
            mlflow.set_tracking_uri(tracking_uri)
            if mlflow.get_experiment_by_name(experiment) is None:
                # Explicit artifact location; otherwise MLflow writes ./mlruns relative to the cwd.
                mlflow.create_experiment(experiment, artifact_location=artifact_uri)
            mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None) -> Iterator[str | None]:
        with self._mlflow.start_run(run_name=name, tags=tags) as active:
            yield active.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        self._mlflow.log_metrics({f"{prefix}{k}": v for k, v in metrics.items()})

    def log_dict(self, payload: dict[str, Any], filename: str) -> None:
        self._mlflow.log_dict(payload, filename)

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def log_model(self, pipeline: Pipeline, example: Any) -> None:
        from mlflow.models import infer_signature

        # Signature from churn *probabilities* (float), not integer class labels.
        signature = infer_signature(example, pipeline.predict_proba(example)[:, POSITIVE_CLASS_INDEX])
        self._mlflow.sklearn.log_model(
            pipeline,
            artifact_path=self.MODEL_ARTIFACT_PATH,
            input_example=example,
            signature=signature,
            pip_requirements=self.pip_requirements,
        )

    def register(self, run_id: str) -> str | None:
        uri = f"runs:/{run_id}/{self.MODEL_ARTIFACT_PATH}"
        # The model registry store initialises lazily here and can reconfigure logging too.
        with preserve_logging():
            version = self._mlflow.register_model(uri, self.registered_model).version
            self._mlflow.MlflowClient().set_registered_model_alias(
                self.registered_model, self.champion_alias, version
            )
        logger.info("Registered %s v%s as '%s'", self.registered_model, version,
                    self.champion_alias)
        return str(version)
