"""Download the champion and challenger bundles from their MLflow aliases.

Run with ``make ab-fetch`` before serving with ``ab_test.enabled: true``. Fetching is a
separate step so that MLflow stays out of the serving runtime.
"""

from __future__ import annotations

from pathlib import Path

from churnai.api.ab import VARIANT_A, VARIANT_B
from churnai.config import Settings
from churnai.logging_utils import get_logger
from churnai.models.tracking import MlflowTracker, make_tracker
from churnai.pipelines._cli import exit_with, run_cli

logger = get_logger(__name__)


def run(settings: Settings, tracker: MlflowTracker | None = None) -> dict[str, Path]:
    tracker = tracker or make_tracker(settings)
    aliases = {VARIANT_A: settings.mlflow.champion_alias, VARIANT_B: settings.mlflow.challenger_alias}
    paths = {v: tracker.fetch_bundle(alias, settings.ab_bundle_path(v)) for v, alias in aliases.items()}
    logger.info("Fetched A/B bundles: %s", {v: str(p) for v, p in paths.items()})
    return paths


def main() -> None:
    exit_with(run_cli("Download the A/B bundles from their MLflow aliases.", run))


if __name__ == "__main__":
    main()
