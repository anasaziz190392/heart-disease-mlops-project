import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_prep import clean, load_raw, COLUMNS, RAW_PATH  # noqa: E402


@pytest.fixture(scope="module")
def raw_df():
    return load_raw(RAW_PATH)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean(raw_df)


def test_raw_file_exists():
    assert os.path.exists(RAW_PATH), "Raw dataset file is missing"


def test_raw_has_expected_columns(raw_df):
    assert list(raw_df.columns) == COLUMNS


def test_raw_has_expected_row_count(raw_df):
    # UCI Cleveland processed dataset has 303 instances
    assert len(raw_df) == 303


def test_clean_has_no_missing_values(clean_df):
    assert clean_df.isna().sum().sum() == 0


def test_target_is_binary(clean_df):
    assert set(clean_df["target"].unique()).issubset({0, 1})


def test_clean_dtypes_numeric(clean_df):
    for col in clean_df.columns:
        assert pd.api.types.is_numeric_dtype(clean_df[col]), f"{col} is not numeric"


def test_no_duplicate_rows_dropped_incorrectly(clean_df):
    # sanity: row count should not exceed raw row count
    assert len(clean_df) <= 303


def test_categorical_columns_are_ints(clean_df):
    for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
        assert clean_df[col].dtype == int or clean_df[col].dtype.kind in "iu"
