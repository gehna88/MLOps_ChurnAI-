"""Phase 1 pipeline: raw CSV -> validated, leakage-free, stratified train/test split.

Run with ``make data`` or ``python -m churnai.pipelines.prepare_data``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from churnai.config import Settings
from churnai.data import io
from churnai.data.preparation import (
    assert_no_leakage,
    drop_columns,
    select_model_columns,
    stratified_split,
)
from churnai.data.schema import TARGET
from churnai.data.validation import DataValidator
from churnai.exceptions import DataValidationError
from churnai.logging_utils import get_logger
from churnai.pipelines._cli import exit_with, run_cli

logger = get_logger(__name__)

ID_KEY = "CustomerId"


@dataclass(frozen=True)
class PreparationSummary:
    raw_rows: int
    rejected_rows: int
    train_rows: int
    test_rows: int
    train_churn_rate: float
    test_churn_rate: float
    dropped_columns: list[str]


def run(settings: Settings) -> PreparationSummary:
    raw = io.read_csv(settings.resolve(settings.paths.raw_data))
    logger.info("Loaded raw data: %d rows x %d columns", *raw.shape)

    unique_key = ID_KEY if ID_KEY in raw.columns else None
    result = DataValidator(require_target=True, unique_key=unique_key).validate(raw)
    if result.rejected_fraction > settings.data.max_rejected_fraction:
        raise DataValidationError(
            f"{result.rejected_fraction:.1%} of rows failed validation "
            f"(limit {settings.data.max_rejected_fraction:.1%}): {result.failure_counts}"
        )
    if not result.rejected.empty:
        io.write_csv(result.rejected, settings.resolve(settings.paths.rejected_dir) / "rejected.csv")

    dropped = [*settings.data.identifier_columns, *settings.data.leakage_columns]
    model_df = select_model_columns(drop_columns(result.valid, dropped))
    assert_no_leakage(model_df, dropped)

    split = stratified_split(model_df, settings.data.test_size, settings.project.random_seed)
    io.write_csv(split.train, settings.train_path)
    io.write_csv(split.test, settings.test_path)

    summary = PreparationSummary(
        raw_rows=len(raw),
        rejected_rows=len(result.rejected),
        train_rows=len(split.train),
        test_rows=len(split.test),
        train_churn_rate=round(float(split.train[TARGET].mean()), 4),
        test_churn_rate=round(float(split.test[TARGET].mean()), 4),
        dropped_columns=dropped,
    )
    io.write_json(asdict(summary), settings.processed_dir / "summary.json")
    logger.info("Data preparation complete: %s", asdict(summary))
    return summary


def main() -> None:
    exit_with(run_cli("Validate raw data and write train/test splits.", run))


if __name__ == "__main__":
    main()
