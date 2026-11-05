import logging
import sys
from pathlib import Path

import pytest

from churnai.models.tracking import MlflowTracker, applicable_requirements
from tests.conftest import REPO_ROOT


def test_requirements_filtered_by_platform_markers() -> None:
    requirements = applicable_requirements(REPO_ROOT / "requirements" / "serve.txt")
    names = {r.split("==")[0].split("[")[0] for r in requirements}
    assert "fastapi" in names
    assert not any(";" in r for r in requirements), "markers must be stripped"
    if sys.platform == "linux":
        assert "xgboost-cpu" in names and "xgboost" not in names
    else:
        assert "xgboost" in names and "xgboost-cpu" not in names


@pytest.mark.slow
def test_fresh_mlflow_database_keeps_our_logging(tmp_path: Path) -> None:
    """Creating a new MLflow DB must not silence the pipeline's INFO logs (seen in CI)."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    project_logger = logging.getLogger("churnai.pipelines.train")

    MlflowTracker(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        artifact_uri=(tmp_path / "artifacts").as_uri(),
        experiment="logging-check",
        registered_model="logging-check-model",
        champion_alias="champion",
    )

    assert root.level == level
    assert root.handlers == handlers
    assert not project_logger.disabled
