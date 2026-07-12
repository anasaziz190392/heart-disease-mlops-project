import json
import os
import sys

import joblib
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features import ALL_FEATURES  # noqa: E402

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "..", "models", "model.pkl")
MODEL_CARD_PATH = os.path.join(HERE, "..", "models", "model_card.json")

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Trained model not found; run src/train.py first")
    return joblib.load(MODEL_PATH)


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "models/model.pkl missing; run src/train.py"


def test_model_card_exists_and_has_metrics():
    assert os.path.exists(MODEL_CARD_PATH)
    with open(MODEL_CARD_PATH) as f:
        card = json.load(f)
    assert "test_metrics" in card
    assert card["test_metrics"]["roc_auc"] > 0.5  # better than random


def test_model_predicts_single_sample(model):
    df = pd.DataFrame([SAMPLE_PATIENT])[ALL_FEATURES]
    pred = model.predict(df)
    assert pred[0] in (0, 1)


def test_model_predict_proba_shape(model):
    df = pd.DataFrame([SAMPLE_PATIENT])[ALL_FEATURES]
    proba = model.predict_proba(df)
    assert proba.shape == (1, 2)
    assert abs(proba.sum() - 1.0) < 1e-6


def test_model_meets_minimum_quality_bar():
    with open(MODEL_CARD_PATH) as f:
        card = json.load(f)
    # Production readiness gate: model must beat a reasonable baseline
    assert card["test_metrics"]["roc_auc"] >= 0.80
    assert card["test_metrics"]["accuracy"] >= 0.75
