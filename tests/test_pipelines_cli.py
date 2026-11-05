"""Tests for the profiling pipeline and the shared CLI wrapper."""

from pathlib import Path

import pytest

from churnai.config import Settings
from churnai.exceptions import DataValidationError
from churnai.pipelines import profile_data
from churnai.pipelines._cli import EXIT_FAILURE, EXIT_OK, run_cli


def test_profile_report_documents_leakage_and_segments(project: Settings) -> None:
    report = profile_data.run(project).read_text(encoding="utf-8")

    assert report.startswith("# ChurnAI data profile")
    complain_row = next(line for line in report.splitlines() if line.startswith("| Complain"))
    assert "LEAKAGE SUSPECTED" in complain_row
    for section in ("Mean of numeric features by outcome", "Churn rate by Geography",
                    "Churn rate by Gender", "Churn rate by Card Type"):
        assert f"## {section}" in report


def test_profile_excludes_identifier_columns(project: Settings) -> None:
    report = profile_data.run(project).read_text(encoding="utf-8")
    for identifier in project.data.identifier_columns:
        assert f"| {identifier} |" not in report


def _config_arg(settings: Settings) -> list[str]:
    return ["--config", str(settings.root / "configs" / "config.yaml")]


def test_cli_returns_ok_on_success(project: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project.root)  # settings root defaults to the cwd
    calls: list[Path] = []
    code = run_cli("test", lambda s: calls.append(s.root), argv=_config_arg(project))
    assert code == EXIT_OK
    assert calls == [project.root]


def test_cli_maps_expected_errors_to_failure_code(
    project: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(project.root)

    def failing(_: Settings) -> None:
        raise DataValidationError("broken upstream export")

    assert run_cli("test", failing, argv=_config_arg(project)) == EXIT_FAILURE


def test_cli_lets_unexpected_bugs_surface(project: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Genuine programming errors must not be swallowed into a quiet exit code."""
    monkeypatch.chdir(project.root)

    def buggy(_: Settings) -> None:
        raise ZeroDivisionError

    with pytest.raises(ZeroDivisionError):
        run_cli("test", buggy, argv=_config_arg(project))
