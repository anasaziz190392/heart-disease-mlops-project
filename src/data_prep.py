"""
data_prep.py
------------!
Loads the raw UCI Cleveland Heart Disease file, assigns column names,
handles missing values, fixes dtypes, binarizes the target, and writes
a clean CSV to data/processed/heart_clean.csv.

The raw file has NO header and uses '?' for missing values.
Column order (14 attributes actually used, per heart-disease.names):
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
    oldpeak, slope, ca, thal, target
"""
import os

import numpy as np
import pandas as pd

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "processed.cleveland.data")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "heart_clean.csv")


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Read the headerless, comma-separated UCI file."""
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, fix dtypes, binarize target."""
    df = df.copy()

    # 'ca' and 'thal' are the only columns with missing values in this dataset
    # (6 rows total). Impute with the column mode since they are categorical/
    # ordinal codes, not continuous measurements.
    for col in ["ca", "thal"]:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_val)

    # Ensure numeric dtypes
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any remaining rows with unexpected NaNs (safety net)
    df = df.dropna().reset_index(drop=True)

    # Cast categorical/ordinal columns to int codes
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(int)

    # Original target is 0 (no disease) .. 4 (varying severity).
    # Per UCI documentation, experiments binarize it: 0 = absence, 1 = presence.
    df["target"] = (df["target"] > 0).astype(int)

    return df


def run(raw_path: str = RAW_PATH, out_path: str = PROCESSED_PATH) -> pd.DataFrame:
    df = load_raw(raw_path)
    df_clean = clean(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_clean.to_csv(out_path, index=False)
    print(f"[data_prep] Cleaned dataset written to {out_path}  shape={df_clean.shape}")
    return df_clean


if __name__ == "__main__":
    run()
