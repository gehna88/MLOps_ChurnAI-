"""Generate ``reports/data_profile.md`` from the raw dataset.

Run with ``make profile`` or ``python -m churnai.pipelines.profile_data``.
"""

from __future__ import annotations

from pathlib import Path

from churnai.config import Settings
from churnai.data import io
from churnai.data.profiling import leakage_scan, render_markdown
from churnai.data.schema import NUMERIC_COLUMNS
from churnai.logging_utils import get_logger
from churnai.pipelines._cli import exit_with, run_cli

logger = get_logger(__name__)


def run(settings: Settings) -> Path:
    df = io.read_csv(settings.resolve(settings.paths.raw_data))
    exclude = list(settings.data.identifier_columns)

    for finding in leakage_scan(df, exclude):
        if finding.suspicious:
            logger.warning("Possible target leakage in '%s' (AUC=%.4f)", finding.column, finding.auc)

    report_path = settings.resolve(settings.paths.reports_dir) / "data_profile.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(df, exclude, list(NUMERIC_COLUMNS)), encoding="utf-8")
    logger.info("Wrote data profile to %s", report_path)
    return report_path


def main() -> None:
    exit_with(run_cli("Profile the raw dataset and scan for leakage.", run))


if __name__ == "__main__":
    main()
