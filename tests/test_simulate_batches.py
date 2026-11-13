import pandas as pd
import pytest

from churnai.config import Settings
from churnai.data.schema import FEATURE_COLUMNS, TARGET
from churnai.pipelines import prepare_data, simulate_batches


@pytest.fixture
def batches(project: Settings) -> dict[str, pd.DataFrame]:
    prepare_data.run(project)
    simulate_batches.run(project)
    out = project.resolve(project.drift_simulation.batch_dir)
    return {year.name: pd.read_csv(out / f"{year.name}.csv") for year in project.drift_simulation.years}


def test_batches_have_model_columns_and_labels(batches: dict[str, pd.DataFrame]) -> None:
    for batch in batches.values():
        assert list(batch.columns) == [*FEATURE_COLUMNS, TARGET]
        assert set(batch[TARGET].unique()) <= {0, 1}


def test_neutral_year_is_the_test_split_unchanged(project: Settings, batches: dict) -> None:
    test = pd.read_csv(project.test_path)
    holdout = batches["2026_holdout"]
    key = sorted(holdout.columns)

    def canonical(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[key].sort_values(key).reset_index(drop=True)

    pd.testing.assert_frame_equal(canonical(holdout), canonical(test))


def test_only_unseen_customers_are_used(project: Settings, batches: dict) -> None:
    train = pd.read_csv(project.train_path)
    key = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts"]
    for batch in batches.values():
        assert batch[key].merge(train[key].drop_duplicates(), on=key).empty


def test_later_years_drift_further(batches: dict[str, pd.DataFrame]) -> None:
    base, mid, late = batches["2026_holdout"], batches["2027"], batches["2028"]
    for column, stat in [("Age", "mean"), ("EstimatedSalary", "mean")]:
        assert getattr(base[column], stat)() < getattr(mid[column], stat)() < getattr(late[column], stat)()
    germany = [(b["Geography"] == "Germany").mean() for b in (base, mid, late)]
    assert germany == sorted(germany)
    assert late[TARGET].mean() > base[TARGET].mean()


def test_simulation_is_reproducible(project: Settings, batches: dict) -> None:
    simulate_batches.run(project)
    out = project.resolve(project.drift_simulation.batch_dir)
    for name, batch in batches.items():
        pd.testing.assert_frame_equal(batch, pd.read_csv(out / f"{name}.csv"))
