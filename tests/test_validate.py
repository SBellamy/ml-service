import pandas as pd
import pytest

from pipeline.validate import validate_df, REQUIRED_COLUMNS


def make_good_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [25, 52000, 1200, 14, 0, 0],
            [42, 88000, 5400, 33, 1, 1],
            [31, 61000, 2300, 18, 0, 0],
            [58, 99000, 15000, 41, 1, 1],
        ],
        columns=REQUIRED_COLUMNS,
    )


def test_validate_df_happy_path():
    df = make_good_df()
    validate_df(df)  # should not raise


def test_validate_df_missing_column_raises():
    df = make_good_df().drop(columns=["income"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_df(df)


def test_validate_df_null_raises():
    df = make_good_df()
    df.loc[0, "age"] = None
    with pytest.raises(ValueError, match="Null values detected"):
        validate_df(df)


def test_validate_df_non_binary_target_raises():
    df = make_good_df()
    df.loc[0, "target"] = 2
    with pytest.raises(ValueError, match="Target column must be binary"):
        validate_df(df)