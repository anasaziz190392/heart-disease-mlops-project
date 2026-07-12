"""
download_data.py
-----------------
Acquisition script for the UCI Heart Disease (Cleveland) dataset.

Usage:
    python data/download_data.py

Behaviour:
    1. Tries to download `processed.cleveland.data` directly from the UCI
       Machine Learning Repository.
    2. If there is no internet access (e.g. offline grading environment),
       it falls back to a local copy that must be placed at
       `data/raw/processed.cleveland.data` (this is the same file shipped
       inside the official `heart_disease.zip` archive referenced in the
       assignment).

Output:
    data/raw/processed.cleveland.data   (untouched raw file, 14 columns, '?' = missing)
"""
import os
import shutil
import sys
import urllib.request

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
RAW_FILE = os.path.join(RAW_DIR, "processed.cleveland.data")

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# Fallback: the zip file distributed with the assignment (already extracted
# by the grader/student into this same folder before running the script).
LOCAL_FALLBACK = os.path.join(RAW_DIR, "processed.cleveland.data")


def download() -> str:
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(RAW_FILE) and os.path.getsize(RAW_FILE) > 0:
        print(f"[download_data] Raw file already present at {RAW_FILE}, skipping download.")
        return RAW_FILE

    try:
        print(f"[download_data] Attempting download from {UCI_URL} ...")
        urllib.request.urlretrieve(UCI_URL, RAW_FILE)
        print("[download_data] Download successful.")
        return RAW_FILE
    except Exception as exc:  # noqa: BLE001
        print(f"[download_data] Download failed ({exc}).")
        if os.path.exists(LOCAL_FALLBACK):
            print("[download_data] Using local fallback copy shipped with the repo.")
            return LOCAL_FALLBACK
        print(
            "[download_data] ERROR: no internet access and no local fallback found.\n"
            "Please unzip heart_disease.zip (from the UCI repository) and place "
            "'processed.cleveland.data' inside data/raw/ manually."
        )
        sys.exit(1)


if __name__ == "__main__":
    path = download()
    print(f"[download_data] Raw dataset ready at: {path}")
