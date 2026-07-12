import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")

if not os.path.exists(MODEL_PATH):
    pytest.skip("Trained model not found; run src/train.py first", allow_module_level=True)

from main import app  # noqa: E402

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid_payload():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["risk_label"] in ("High Risk", "Low Risk")


def test_predict_missing_field_returns_422():
    bad_payload = VALID_PAYLOAD.copy()
    del bad_payload["age"]
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_invalid_range_returns_422():
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["sex"] = 5  # out of allowed range 0-1
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"predict_requests_total" in resp.content or resp.status_code == 200
