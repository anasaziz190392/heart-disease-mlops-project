import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features import ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_preprocessor  # noqa: E402
from data_prep import PROCESSED_PATH  # noqa: E402


@pytest.fixture(scope="module")
def sample_df():
    if not os.path.exists(PROCESSED_PATH):
        pytest.skip("Processed dataset not found; run src/data_prep.py first")
    return pd.read_csv(PROCESSED_PATH)


def test_feature_lists_disjoint():
    assert set(NUMERIC_FEATURES).isdisjoint(set(CATEGORICAL_FEATURES))


def test_all_features_cover_expected_count():
    assert len(ALL_FEATURES) == 13  # 14 columns minus target


def test_preprocessor_builds():
    preprocessor = build_preprocessor()
    assert preprocessor is not None


def test_preprocessor_fits_and_transforms(sample_df):
    preprocessor = build_preprocessor()
    X = sample_df[ALL_FEATURES]
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(sample_df)
    # scaled numeric + one-hot categorical -> more columns than raw features
    assert transformed.shape[1] >= len(ALL_FEATURES)


def test_preprocessor_handles_unseen_category(sample_df):
    preprocessor = build_preprocessor()
    X = sample_df[ALL_FEATURES]
    preprocessor.fit(X)
    unseen = X.iloc[[0]].copy()
    unseen["thal"] = 999  # category never seen during fit
    # should not raise because handle_unknown="ignore"
    out = preprocessor.transform(unseen)
    assert out.shape[0] == 1
