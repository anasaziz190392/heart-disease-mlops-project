"""
train.py
--------
End-to-end training pipeline for the Heart Disease classifier.

Steps:
  1. Load cleaned data (data/processed/heart_clean.csv)
  2. Train/test split (stratified)
  3. Build preprocessing + model pipelines for:
        - Logistic Regression  (GridSearchCV over C, penalty)
        - Random Forest        (GridSearchCV over n_estimators, max_depth)
  4. 5-fold cross-validation on the training set
  5. Evaluate on held-out test set: accuracy, precision, recall, F1, ROC-AUC
  6. Log params/metrics to MLflow
  7. Persist the best overall model (pipeline incl. preprocessing) to
     models/model.pkl for serving, plus models/model_card.json with metadata.

Usage:
    python src/train.py
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(__file__))
from features import ALL_FEATURES, TARGET, build_preprocessor  # noqa: E402
from mlflow_compat import mlflow  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "processed", "heart_clean.csv")
MODEL_DIR = os.path.join(HERE, "..", "models")
FIG_DIR = os.path.join(HERE, "..", "reports", "figures")
RANDOM_STATE = 42

MODEL_GRIDS = {
    "LogisticRegression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 4, 6, 8],
            "clf__min_samples_leaf": [1, 2, 4],
        },
    },
}


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }, y_pred, y_proba


def run():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Configure MLflow to use a local backend directory inside /app
    mlflow_dir = os.path.join(HERE, "..", "mlruns")
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")

    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_experiment("heart_disease_classification")

    results_summary = []
    best_overall = {"score": -1, "model": None, "name": None, "metrics": None}

    for name, cfg in MODEL_GRIDS.items():
        with mlflow.start_run(run_name=name):
            preprocessor = build_preprocessor()
            pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", cfg["estimator"])])

            grid = GridSearchCV(
                pipe, cfg["param_grid"], cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
            )
            grid.fit(X_train, y_train)
            best_pipe = grid.best_estimator_

            cv_scores = grid.cv_results_["mean_test_score"]
            metrics, y_pred, y_proba = evaluate(best_pipe, X_test, y_test)

            # ---- MLflow logging ----
            mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})
            mlflow.log_param("model_type", name)
            mlflow.log_metric("cv_best_roc_auc", float(grid.best_score_))
            mlflow.log_metrics(metrics)

            # Confusion matrix plot (save locally only, skip artifact logging)
            fig, ax = plt.subplots(figsize=(5, 4.5))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, display_labels=["No Disease", "Disease"], cmap="Blues", ax=ax
            )
            ax.set_title(f"Confusion Matrix - {name}", fontweight="bold")
            fig.tight_layout()
            cm_path = os.path.join(FIG_DIR, f"confusion_matrix_{name}.png")
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)

            # ROC curve plot (save locally only, skip artifact logging)
            fig, ax = plt.subplots(figsize=(5, 4.5))
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)
            ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
            ax.set_title(f"ROC Curve - {name}", fontweight="bold")
            fig.tight_layout()
            roc_path = os.path.join(FIG_DIR, f"roc_curve_{name}.png")
            fig.savefig(roc_path, dpi=150)
            plt.close(fig)

            print(f"[train] {name}: best_params={grid.best_params_}")
            print(f"[train] {name}: cv_roc_auc={grid.best_score_:.4f}  test_metrics={metrics}")

            results_summary.append({"model": name, "cv_roc_auc": float(grid.best_score_), **metrics,
                                     "best_params": grid.best_params_})

            if metrics["roc_auc"] > best_overall["score"]:
                best_overall = {"score": metrics["roc_auc"], "model": best_pipe, "name": name, "metrics": metrics}

    # ---- Persist best model for serving ----
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(best_overall["model"], model_path)

    model_card = {
        "best_model": best_overall["name"],
        "test_metrics": best_overall["metrics"],
        "features": ALL_FEATURES,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "sklearn_pipeline": True,
    }
    with open(os.path.join(MODEL_DIR, "model_card.json"), "w") as f:
        json.dump(model_card, f, indent=2)

    with open(os.path.join(HERE, "..", "reports", "training_results.json"), "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"\n[train] BEST MODEL: {best_overall['name']} (test ROC-AUC={best_overall['score']:.4f})")
    print(f"[train] Saved to {model_path}")


if __name__ == "__main__":
    run()
