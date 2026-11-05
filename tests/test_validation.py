import pandas as pd
import pytest

from churnai.data.validation import REJECTION_REASON_COLUMN, DataValidator
from churnai.exceptions import DataValidationError


@pytest.fixture
def small(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.head(50).copy()


def test_real_data_passes_validation(raw_df: pd.DataFrame) -> None:
    result = DataValidator(unique_key="CustomerId").validate(raw_df)
    assert result.rejected.empty
    assert len(result.valid) == len(raw_df)


@pytest.mark.parametrize(
    ("column", "bad_value", "reason"),
    [
        ("Age", 5, "Age:out_of_range"),
        ("CreditScore", 2000, "CreditScore:out_of_range"),
        ("Geography", "Atlantis", "Geography:unknown_category"),
        ("NumOfProducts", 2.5, "NumOfProducts:not_integer"),
        ("Balance", "lots", "Balance:null"),  # junk coerced to NaN
        ("Exited", 3, "Exited:invalid"),
    ],
)
def test_bad_rows_are_rejected_with_reason(
    small: pd.DataFrame, column: str, bad_value: object, reason: str
) -> None:
    small[column] = small[column].astype(object)
    small.loc[0, column] = bad_value
    result = DataValidator().validate(small)
    assert len(result.rejected) == 1
    assert reason in result.rejected[REJECTION_REASON_COLUMN].iloc[0]
    assert len(result.valid) == len(small) - 1


def test_duplicate_ids_rejected(small: pd.DataFrame) -> None:
    duplicated = pd.concat([small, small.head(2)], ignore_index=True)
    result = DataValidator(unique_key="CustomerId").validate(duplicated)
    assert len(result.rejected) == 2
    assert result.rejected[REJECTION_REASON_COLUMN].str.contains("duplicate").all()


def test_missing_column_raises(small: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="Missing required columns"):
        DataValidator().validate(small.drop(columns=["Age"]))


def test_target_optional_for_inference(small: pd.DataFrame) -> None:
    result = DataValidator(require_target=False).validate(small.drop(columns=["Exited"]))
    assert result.rejected.empty
