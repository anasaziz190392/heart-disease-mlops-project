"""
Automated model retraining pipeline triggered by drift detection.
Includes model evaluation, versioning, and promotion logic.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.features import ALL_FEATURES, TARGET, build_preprocessor
from src.model_registry import register_model, promote_model
from src.drift_detection import detect_prediction_drift, store_drift_report

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path("models")
DATA_PATH = Path("data/processed/heart_clean.csv")


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression model."""
    preprocessor = build_preprocessor()
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, random_state=42))
    ])
    
    param_grid = {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    metrics = evaluate_model(grid.best_estimator_, X_test, y_test)
    return grid.best_estimator_, metrics, grid.best_params_


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model."""
    preprocessor = build_preprocessor()
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 4, 6, 8],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    metrics = evaluate_model(grid.best_estimator_, X_test, y_test)
    return grid.best_estimator_, metrics, grid.best_params_


def retrain_models(check_performance_threshold: float = 0.05):
    """
    Retrain models and promote if performance is better.
    
    Args:
        check_performance_threshold: Minimum accuracy improvement to promote new model
    """
    logger.info("Starting automated retraining pipeline")
    
    # Load data
    if not DATA_PATH.exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        return False
    
    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train both models
    logger.info("Training Logistic Regression...")
    lr_model, lr_metrics, lr_params = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    logger.info("Training Random Forest...")
    rf_model, rf_metrics, rf_params = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Select best model
    best_model_name = "LogisticRegression" if lr_metrics["roc_auc"] > rf_metrics["roc_auc"] else "RandomForest"
    best_model = lr_model if best_model_name == "LogisticRegression" else rf_model
    best_metrics = lr_metrics if best_model_name == "LogisticRegression" else rf_metrics
    
    logger.info(f"Best model: {best_model_name} (ROC-AUC: {best_metrics['roc_auc']:.4f})")
    
    # Register new model version
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"model_{model_version}.pkl"
    
    joblib.dump(best_model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Register in model registry
    register_model(
        str(model_path),
        "heart_disease_classifier",
        version=model_version,
        metrics=best_metrics,
        tags=["retraining", best_model_name, "production"],
    )
    
    # Check if should promote
    current_model_path = MODEL_DIR / "model.pkl"
    if current_model_path.exists():
        current_model = joblib.load(current_model_path)
        current_metrics = evaluate_model(current_model, X_test, y_test)
        
        accuracy_improvement = best_metrics["accuracy"] - current_metrics["accuracy"]
        logger.info(f"Accuracy improvement: {accuracy_improvement:.4f}")
        
        if accuracy_improvement > check_performance_threshold:
            logger.info("New model meets performance threshold. Promoting to production...")
            joblib.dump(best_model, current_model_path)
            promote_model("heart_disease_classifier", model_version)
            logger.info("Model promoted to production")
            return True
        else:
            logger.info("New model does not meet performance threshold. Not promoting.")
            return False
    else:
        # No current model, save as production
        joblib.dump(best_model, current_model_path)
        promote_model("heart_disease_classifier", model_version)
        logger.info("First model saved and promoted to production")
        return True


if __name__ == "__main__":
    success = retrain_models()
    exit(0 if success else 1)
