import sys

from churnai.models.tracking import applicable_requirements
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
