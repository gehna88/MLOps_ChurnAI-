import pandas as pd
import pytest

from churnai.config import Settings
from churnai.data.preparation import assert_no_leakage, select_model_columns, stratified_split
from churnai.data.profiling import leakage_scan
from churnai.data.schema import FEATURE_COLUMNS, TARGET
from churnai.exceptions import DataValidationError
from churnai.pipelines import prepare_data
from tests.conftest import make_project

FORBIDDEN = ["Complain", "RowNumber", "CustomerId", "Surname"]
CHURN_RATE_TOLERANCE = 0.01


def test_prepare_writes_leakage_free_splits(project: Settings) -> None:
    summary = prepare_data.run(project)
    train = pd.read_csv(project.train_path)
    test = pd.read_csv(project.test_path)

    for frame in (train, test):
        assert not set(FORBIDDEN) & set(frame.columns), "leaky/identifier column reached training"
        assert list(frame.columns) == [*FEATURE_COLUMNS, TARGET]

    assert summary.train_rows + summary.test_rows == summary.raw_rows - summary.rejected_rows
    assert abs(summary.train_churn_rate - summary.test_churn_rate) < CHURN_RATE_TOLERANCE


def test_prepare_aborts_when_too_many_rows_invalid(tmp_path, sample_raw_df) -> None:
    broken = sample_raw_df.copy()
    broken["Age"] = 1  # every row out of range
    settings = make_project(tmp_path, broken)
    with pytest.raises(DataValidationError, match="failed validation"):
        prepare_data.run(settings)


def test_allow_list_excludes_unknown_new_columns(sample_raw_df: pd.DataFrame) -> None:
    df = sample_raw_df.assign(SomeNewLeakyColumn=sample_raw_df[TARGET])
    assert "SomeNewLeakyColumn" not in select_model_columns(df).columns


def test_assert_no_leakage_raises() -> None:
    with pytest.raises(DataValidationError, match="Complain"):
        assert_no_leakage(pd.DataFrame({"Complain": [1]}), ["Complain"])


def test_stratified_split_is_deterministic(sample_raw_df: pd.DataFrame) -> None:
    a = stratified_split(sample_raw_df, 0.2, seed=1)
    b = stratified_split(sample_raw_df, 0.2, seed=1)
    pd.testing.assert_frame_equal(a.test, b.test)


def test_profiling_flags_complain_as_leakage(raw_df: pd.DataFrame) -> None:
    findings = {f.column: f for f in leakage_scan(raw_df, ["RowNumber", "CustomerId", "Surname"])}
    assert findings["Complain"].suspicious
    assert not any(findings[c].suspicious for c in FEATURE_COLUMNS)
