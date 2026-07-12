"""
inference.py
-------------
Standalone batch inference script — loads the persisted pipeline
(preprocessing + model bundled together) and scores new patient records
from a CSV, without needing the API running.

Usage:
    python src/inference.py --input new_patients.csv --output predictions.csv
    python src/inference.py --single  # runs one example record for a smoke test
"""
import argparse
import json
import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from features import ALL_FEATURES  # noqa: E402

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "..", "models", "model.pkl")

EXAMPLE_RECORD = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run `python src/train.py` first.")
    return joblib.load(path)


def predict_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    missing = set(ALL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    X = df[ALL_FEATURES]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["prediction"] = preds
    out["risk_label"] = out["prediction"].map({1: "High Risk", 0: "Low Risk"})
    out["confidence"] = probs
    return out


def main():
    parser = argparse.ArgumentParser(description="Batch inference for heart disease risk model")
    parser.add_argument("--input", type=str, help="Path to input CSV with patient records")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to write predictions CSV")
    parser.add_argument("--single", action="store_true", help="Run a single example record smoke test")
    args = parser.parse_args()

    model = load_model()

    if args.single or not args.input:
        df = pd.DataFrame([EXAMPLE_RECORD])
        result = predict_dataframe(model, df)
        print(json.dumps(result.to_dict(orient="records")[0], indent=2, default=str))
        return

    df = pd.read_csv(args.input)
    result = predict_dataframe(model, df)
    result.to_csv(args.output, index=False)
    print(f"[inference] Wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
