"""
app/main.py
-----------
FastAPI service exposing the trained Heart Disease risk model.

Endpoints:
    GET  /health            -> liveness/readiness probe with dependency checks
    GET  /metrics           -> Prometheus metrics
    POST /token             -> Generate JWT access token
    POST /predict           -> Predict heart disease risk (requires JWT token)
    GET  /explain/<int>     -> Get SHAP explanation for a prediction

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import json
import logging
import os
import time
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import Response
from fastapi.security import HTTPBearer
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from app.auth import verify_token, generate_user_token, TokenData, Token
from app.validation import sanitize_patient_data

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart_disease_api")

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
HERE = os.path.dirname(__file__)
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(HERE, "..", "models", "model.pkl"))

model = None
model_load_error = None
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as exc:  # noqa: BLE001
    model_load_error = str(exc)
    logger.error(f"Failed to load model from {MODEL_PATH}: {exc}")

# --------------------------------------------------------------------------
# Prometheus metrics
# --------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "predict_requests_total", "Total number of /predict requests", ["status"]
)
REQUEST_LATENCY = Histogram(
    "predict_request_latency_seconds", "Latency of /predict requests in seconds"
)
PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Total predictions made", ["prediction"]
)

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts the risk of heart disease from patient health data with JWT authentication.",
    version="2.0.0",
)


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------
class PatientData(BaseModel):
    age: float = Field(..., example=63, description="Age in years")
    sex: int = Field(..., ge=0, le=1, example=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, example=1, description="Chest pain type (1-4)")
    trestbps: float = Field(..., example=145, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., example=233, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, example=1, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: int = Field(..., ge=0, le=2, example=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., example=150, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, example=0, description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., example=2.3, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, example=3, description="Slope of peak exercise ST segment (1-3)")
    ca: int = Field(..., ge=0, le=3, example=0, description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: int = Field(..., ge=3, le=7, example=6, description="Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)")


class PredictionResponse(BaseModel):
    prediction: int
    risk_label: str
    confidence: float
    message: str = "Prediction successful"


class CredentialsRequest(BaseModel):
    username: str
    password: str


# --------------------------------------------------------------------------
# Middleware for request logging + latency metric
# --------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        f"{request.method} {request.url.path} status={response.status_code} "
        f"duration={duration:.4f}s client={request.client.host if request.client else 'unknown'}"
    )
    return response


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Heart Disease Risk Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "token": "POST /token (requires username & password)",
        "predict": "POST /predict (requires JWT token)",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    """Enhanced healthcheck with dependency validation."""
    health_status = {
        "status": "ok",
        "model_loaded": model is not None,
        "dependencies": {},
    }
    
    # Check model file
    if model is None:
        health_status["status"] = "degraded"
        health_status["dependencies"]["model"] = f"failed: {model_load_error}"
    else:
        health_status["dependencies"]["model"] = "ok"
    
    # Check data preprocessing
    try:
        from src.features import build_preprocessor
        build_preprocessor()
        health_status["dependencies"]["preprocessing"] = "ok"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["dependencies"]["preprocessing"] = f"error: {str(e)}"
    
    # Test a dummy prediction
    try:
        if model is not None:
            dummy_data = pd.DataFrame([{
                "age": 50, "sex": 1, "cp": 1, "trestbps": 120, "chol": 200,
                "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 1
            }])
            model.predict(dummy_data)
            health_status["dependencies"]["prediction_engine"] = "ok"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["dependencies"]["prediction_engine"] = f"error: {str(e)}"
    
    if health_status["status"] == "degraded":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/token", response_model=Token)
def login_for_access_token(credentials: CredentialsRequest):
    """Generate JWT token for authenticated users."""
    try:
        token = generate_user_token(credentials.username)
        logger.info(f"Token generated for user: {credentials.username}")
        return token
    except HTTPException as e:
        logger.warning(f"Failed token request for user: {credentials.username}")
        raise


@app.post("/predict", response_model=PredictionResponse)
def predict(
    patient: PatientData,
    token_data: TokenData = Depends(verify_token),
):
    """Predict heart disease risk (requires JWT token)."""
    if model is None:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=f"Model not loaded: {model_load_error}")
    
    # Validate input data
    is_valid, error_msg, sanitized_data = sanitize_patient_data(patient.dict())
    if not is_valid:
        REQUEST_COUNT.labels(status="validation_error").inc()
        raise HTTPException(status_code=422, detail=f"Validation error: {error_msg}")

    with REQUEST_LATENCY.time():
        try:
            df = pd.DataFrame([sanitized_data])
            pred = int(model.predict(df)[0])
            proba = float(model.predict_proba(df)[0][pred])
            PREDICTIONS_TOTAL.labels(prediction=pred).inc()
            REQUEST_COUNT.labels(status="success").inc()
            
            logger.info(
                f"Prediction made by {token_data.username}: pred={pred}, "
                f"confidence={proba:.4f}"
            )
            
            return PredictionResponse(
                prediction=pred,
                risk_label="High Risk" if pred == 1 else "Low Risk",
                confidence=round(proba, 4),
            )
        except Exception as exc:  # noqa: BLE001
            REQUEST_COUNT.labels(status="error").inc()
            logger.exception("Prediction failed")
            raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")


@app.get("/explain")
def explain_prediction(
    age: float, sex: int, cp: int, trestbps: float, chol: float,
    fbs: int, restecg: int, thalach: float, exang: int, oldpeak: float,
    slope: int, ca: int, thal: int,
    token_data: TokenData = Depends(verify_token),
):
    """Get SHAP explanation for a prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import shap
        
        patient_data = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }])
        
        # Get base model from pipeline
        base_model = model.named_steps["clf"]
        X_preprocessed = model.named_steps["preprocess"].transform(patient_data)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_preprocessed)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        pred = model.predict(patient_data)[0]
        confidence = model.predict_proba(patient_data)[0][int(pred)]
        
        logger.info(f"SHAP explanation requested by {token_data.username}")
        
        return {
            "prediction": int(pred),
            "confidence": float(confidence),
            "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
            "base_value": float(explainer.expected_value),
            "message": "SHAP explanation computed"
        }
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="SHAP library not installed. Install with: pip install shap"
        )
    except Exception as e:
        logger.exception("SHAP explanation failed")
        raise HTTPException(status_code=400, detail=f"Explanation failed: {str(e)}")
