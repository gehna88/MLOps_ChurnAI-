"""Add SHAP reports to candidate runs that finished before training logged them.

Run with ``make shap-backfill`` or ``python -m churnai.pipelines.backfill_shap``.

Only files are added: each run's params, metrics, model and registry aliases stay as
they are. Runs that already have a report are skipped, so this is safe to run again.
Reports use the current training split, so run it only while that split is the one
those runs were trained on.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from churnai.config import Settings
from churnai.data import io
from churnai.data.schema import FEATURE_COLUMNS
from churnai.explain.report import SHAP_ARTIFACT_PATH, write_shap_report
from churnai.logging_utils import get_logger
from churnai.models.tracking import MlflowTracker, make_tracker
from churnai.pipelines._cli import exit_with, run_cli

logger = get_logger(__name__)


@dataclass(frozen=True)
class BackfillSummary:
    updated: list[str]
    skipped: list[str]


def run(settings: Settings, tracker: MlflowTracker | None = None) -> BackfillSummary:
    tracker = tracker or make_tracker(settings)
    X_train, _ = io.split_features_target(io.read_csv(settings.train_path))
    X_train = X_train[list(FEATURE_COLUMNS)]

    updated: list[str] = []
    skipped: list[str] = []
    for run_id in tracker.model_run_ids():
        if tracker.has_artifacts(run_id, SHAP_ARTIFACT_PATH):
            skipped.append(run_id)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = tracker.load_pipeline(run_id)
            for path in write_shap_report(pipeline, X_train, settings, Path(tmp)):
                tracker.log_artifact_to_run(run_id, path, SHAP_ARTIFACT_PATH)
        updated.append(run_id)
        logger.info("Added SHAP report to run %s", run_id)
    logger.info("SHAP backfill: %d runs updated, %d already had a report", len(updated), len(skipped))
    return BackfillSummary(updated, skipped)


def main() -> None:
    exit_with(run_cli("Add SHAP reports to earlier MLflow candidate runs.", run))


if __name__ == "__main__":
    main()
