"""
features.py
------------
Builds the scikit-learn preprocessing pipeline (scaling + one-hot encoding)
used both at training time and at inference time, guaranteeing full
reproducibility (no train/serve skew).
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Low-cardinality categorical / ordinal codes -> one-hot encode
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"


def build_preprocessor() -> ColumnTransformer:
    """Column-wise transformer: scale numeric, one-hot encode categorical."""
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor
