"""Simulate future years of customers for the drift study.

Run with ``make simulate-batches`` or ``python -m churnai.pipelines.simulate_batches``,
after ``make data``.

Each year is a re-weighted sample of REAL customers from the held-out test split, so
no customer the model was trained on appears and every label is genuine. Weights favour
older, German and inactive customers more each year (covariate shift); churn rises on
its own because those segments really churn more. Salaries are inflated per year. A
year with neutral settings is the test split itself, shuffled.

Output: ``data/batches/<year>.csv`` with the model features and ``Exited``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from churnai.config import DriftYear, Settings
from churnai.data import io
from churnai.data.schema import FEATURE_COLUMNS, TARGET
from churnai.logging_utils import get_logger
from churnai.pipelines._cli import exit_with, run_cli

logger = get_logger(__name__)

GERMANY = "Germany"
SALARY = "EstimatedSalary"
SALARY_DECIMALS = 2


def sampling_weights(pool: pd.DataFrame, year: DriftYear) -> np.ndarray:
    """Relative chance of each customer being drawn into this year's batch."""
    age_z = (pool["Age"] - pool["Age"].mean()) / pool["Age"].std()
    weights = np.exp(year.age_tilt * age_z.to_numpy())
    weights *= np.where(pool["Geography"] == GERMANY, year.germany_weight, 1.0)
    weights *= np.where(pool["IsActiveMember"] == 0, year.inactive_weight, 1.0)
    return weights / weights.sum()


def simulate_year(pool: pd.DataFrame, year: DriftYear, size: int, rng: np.random.Generator) -> pd.DataFrame:
    if year.is_neutral:
        return pool.sample(frac=1.0, random_state=rng.integers(2**32)).reset_index(drop=True)
    rows = rng.choice(len(pool), size=size, replace=True, p=sampling_weights(pool, year))
    batch = pool.iloc[rows].reset_index(drop=True)
    batch[SALARY] = (batch[SALARY] * (1 + year.salary_growth)).round(SALARY_DECIMALS)
    return batch


def run(settings: Settings) -> list[Path]:
    cfg = settings.drift_simulation
    pool = io.read_csv(settings.test_path)[[*FEATURE_COLUMNS, TARGET]]
    rng = np.random.default_rng(settings.project.random_seed)
    out_dir = settings.resolve(cfg.batch_dir)

    written = []
    for year in cfg.years:
        batch = simulate_year(pool, year, cfg.batch_size, rng)
        written.append(io.write_csv(batch, out_dir / f"{year.name}.csv"))
        logger.info(
            "Batch %-13s rows=%d mean_age=%.1f germany=%.0f%% active=%.0f%% churn=%.1f%%",
            year.name, len(batch), batch["Age"].mean(), 100 * (batch["Geography"] == GERMANY).mean(),
            100 * batch["IsActiveMember"].mean(), 100 * batch[TARGET].mean(),
        )
    return written


def main() -> None:
    exit_with(run_cli("Simulate drifted future batches from the held-out test split.", run))


if __name__ == "__main__":
    main()
