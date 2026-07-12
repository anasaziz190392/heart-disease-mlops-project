"""
Model drift detection for continuous monitoring.
Detects data distribution shifts and model performance degradation.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DRIFT_HISTORY_FILE = Path("reports/drift_history.json")


def calculate_feature_statistics(X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate mean and std for each feature."""
    stats = {}
    for col in X.columns:
        stats[col] = {
            "mean": float(X[col].mean()),
            "std": float(X[col].std()),
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "q25": float(X[col].quantile(0.25)),
            "q75": float(X[col].quantile(0.75)),
        }
    return stats


def calculate_kolmogorov_smirnov_stat(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
    """
    Calculate KS statistic between two distributions.
    Returns (ks_statistic, p_value)
    """
    from scipy.stats import ks_2samp
    return ks_2samp(data1, data2)


def detect_covariate_shift(
    reference_X: pd.DataFrame,
    current_X: pd.DataFrame,
    threshold: float = 0.05,
) -> Dict:
    """
    Detect covariate shift (data distribution change) using KS test.
    Returns drift report for each feature.
    """
    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "threshold": threshold,
        "drifted_features": [],
        "features": {},
    }
    
    for col in reference_X.columns:
        if col not in current_X.columns:
            continue
        
        ks_stat, p_value = calculate_kolmogorov_smirnov_stat(
            reference_X[col].values,
            current_X[col].values,
        )
        
        is_drifted = p_value < threshold
        drift_report["features"][col] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "drifted": is_drifted,
        }
        
        if is_drifted:
            drift_report["drifted_features"].append(col)
    
    drift_report["has_drift"] = len(drift_report["drifted_features"]) > 0
    return drift_report


def detect_prediction_drift(
    y_true_reference: np.ndarray,
    y_pred_reference: np.ndarray,
    y_true_current: np.ndarray,
    y_pred_current: np.ndarray,
) -> Dict:
    """
    Detect prediction/label drift by comparing accuracy and label distribution.
    """
    from scipy.stats import chi2_contingency
    
    ref_accuracy = (y_true_reference == y_pred_reference).mean()
    curr_accuracy = (y_true_current == y_pred_current).mean()
    accuracy_drop = ref_accuracy - curr_accuracy
    
    # Chi-square test for label distribution shift
    ref_labels = pd.Series(y_true_reference).value_counts(normalize=True)
    curr_labels = pd.Series(y_true_current).value_counts(normalize=True)
    
    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "reference_accuracy": float(ref_accuracy),
        "current_accuracy": float(curr_accuracy),
        "accuracy_drop": float(accuracy_drop),
        "label_distribution": {
            "reference": ref_labels.to_dict(),
            "current": curr_labels.to_dict(),
        },
        "has_drift": accuracy_drop > 0.05,  # Alert if accuracy drops > 5%
    }
    
    return drift_report


def store_drift_report(report: Dict):
    """Store drift detection report in history."""
    history = []
    if DRIFT_HISTORY_FILE.exists():
        with open(DRIFT_HISTORY_FILE) as f:
            history = json.load(f)
    
    history.append(report)
    history = history[-100:]  # Keep last 100 reports
    
    DRIFT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Drift report stored: {report['timestamp']}")


def get_drift_summary() -> Dict:
    """Get summary of recent drift events."""
    if not DRIFT_HISTORY_FILE.exists():
        return {"no_data": True}
    
    with open(DRIFT_HISTORY_FILE) as f:
        history = json.load(f)
    
    if not history:
        return {"no_data": True}
    
    recent = history[-10:]
    drift_events = [r for r in recent if r.get("has_drift", False)]
    
    return {
        "total_checks": len(history),
        "recent_checks": len(recent),
        "drift_events": len(drift_events),
        "last_check": history[-1]["timestamp"],
        "drifted_in_recent": len(drift_events) > 0,
        "recent_drifted_features": set(
            f for r in drift_events for f in r.get("drifted_features", [])
        ),
    }
