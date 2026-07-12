"""
Input validation and sanitization for patient data.
Prevents injection attacks and ensures data quality.
"""
from typing import Optional, Tuple
from pydantic import validator, BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# Clinical constraints for heart disease data
CLINICAL_CONSTRAINTS = {
    "age": {"min": 0, "max": 150, "description": "Age in years"},
    "sex": {"values": [0, 1], "description": "0=female, 1=male"},
    "cp": {"min": 1, "max": 4, "description": "Chest pain type"},
    "trestbps": {"min": 70, "max": 200, "description": "Resting BP (mm Hg)"},
    "chol": {"min": 100, "max": 600, "description": "Cholesterol (mg/dl)"},
    "fbs": {"values": [0, 1], "description": "Fasting blood sugar"},
    "restecg": {"values": [0, 1, 2], "description": "Resting ECG"},
    "thalach": {"min": 40, "max": 220, "description": "Max heart rate"},
    "exang": {"values": [0, 1], "description": "Exercise induced angina"},
    "oldpeak": {"min": 0, "max": 10, "description": "ST depression"},
    "slope": {"values": [1, 2, 3], "description": "ST slope"},
    "ca": {"values": [0, 1, 2, 3], "description": "Major vessels"},
    "thal": {"values": [3, 6, 7], "description": "Thalassemia"},
}


class ValidatedPatientData(BaseModel):
    """Patient data with clinical validation."""
    age: float = Field(..., description="Age in years")
    sex: int = Field(..., description="1=male, 0=female")
    cp: int = Field(..., description="Chest pain type (1-4)")
    trestbps: float = Field(..., description="Resting blood pressure")
    chol: float = Field(..., description="Serum cholesterol")
    fbs: int = Field(..., description="Fasting blood sugar")
    restecg: int = Field(..., description="Resting ECG")
    thalach: float = Field(..., description="Max heart rate")
    exang: int = Field(..., description="Exercise induced angina")
    oldpeak: float = Field(..., description="ST depression")
    slope: int = Field(..., description="ST slope")
    ca: int = Field(..., description="Major vessels")
    thal: int = Field(..., description="Thalassemia")
    
    @validator("age")
    def validate_age(cls, v):
        constraints = CLINICAL_CONSTRAINTS["age"]
        if not (constraints["min"] <= v <= constraints["max"]):
            raise ValueError(
                f"Age must be between {constraints['min']} and {constraints['max']}, got {v}"
            )
        return v
    
    @validator("sex")
    def validate_sex(cls, v):
        if v not in CLINICAL_CONSTRAINTS["sex"]["values"]:
            raise ValueError(f"Sex must be 0 or 1, got {v}")
        return v
    
    @validator("cp")
    def validate_cp(cls, v):
        constraints = CLINICAL_CONSTRAINTS["cp"]
        if not (constraints["min"] <= v <= constraints["max"]):
            raise ValueError(
                f"Chest pain type must be {constraints['min']}-{constraints['max']}, got {v}"
            )
        return v
    
    @validator("trestbps")
    def validate_trestbps(cls, v):
        constraints = CLINICAL_CONSTRAINTS["trestbps"]
        if not (constraints["min"] <= v <= constraints["max"]):
            raise ValueError(
                f"Resting BP must be {constraints['min']}-{constraints['max']} mm Hg, got {v}"
            )
        return v
    
    @validator("chol")
    def validate_chol(cls, v):
        constraints = CLINICAL_CONSTRAINTS["chol"]
        if not (constraints["min"] <= v <= constraints["max"]):
            raise ValueError(
                f"Cholesterol must be {constraints['min']}-{constraints['max']} mg/dl, got {v}"
            )
        return v
    
    @validator("fbs")
    def validate_fbs(cls, v):
        if v not in CLINICAL_CONSTRAINTS["fbs"]["values"]:
            raise ValueError(f"Fasting blood sugar must be 0 or 1, got {v}")
        return v
    
    @validator("restecg")
    def validate_restecg(cls, v):
        if v not in CLINICAL_CONSTRAINTS["restecg"]["values"]:
            raise ValueError(f"Resting ECG must be 0, 1, or 2, got {v}")
        return v
    
    @validator("thalach")
    def validate_thalach(cls, v):
        constraints = CLINICAL_CONSTRAINTS["thalach"]
        if not (constraints["min"] <= v <= constraints["max"]):
            raise ValueError(
                f"Max heart rate must be {constraints['min']}-{constraints['max']}, got {v}"
            )
        return v
    
    @validator("exang")
    def validate_exang(cls, v):
        if v not in CLINICAL_CONSTRAINTS["exang"]["values"]:
            raise ValueError(f"Exercise angina must be 0 or 1, got {v}")
        return v
    
    @validator("oldpeak")
    def validate_oldpeak(cls, v):
        constraints = CLINICAL_CONSTRAINTS["oldpeak"]
        if not (constraints["min"] <= v <= constraints["max"]):
            raise ValueError(
                f"ST depression must be {constraints['min']}-{constraints['max']}, got {v}"
            )
        return v
    
    @validator("slope")
    def validate_slope(cls, v):
        if v not in CLINICAL_CONSTRAINTS["slope"]["values"]:
            raise ValueError(f"ST slope must be 1, 2, or 3, got {v}")
        return v
    
    @validator("ca")
    def validate_ca(cls, v):
        if v not in CLINICAL_CONSTRAINTS["ca"]["values"]:
            raise ValueError(f"Major vessels must be 0, 1, 2, or 3, got {v}")
        return v
    
    @validator("thal")
    def validate_thal(cls, v):
        if v not in CLINICAL_CONSTRAINTS["thal"]["values"]:
            raise ValueError(f"Thalassemia must be 3, 6, or 7, got {v}")
        return v


def sanitize_patient_data(data: dict) -> Tuple[bool, Optional[str], dict]:
    """
    Sanitize and validate patient data.
    Returns (is_valid, error_message, sanitized_data)
    """
    try:
        validated = ValidatedPatientData(**data)
        logger.info("Patient data validated successfully")
        return True, None, validated.dict()
    except ValueError as e:
        logger.warning(f"Validation failed: {str(e)}")
        return False, str(e), {}
    except Exception as e:
        logger.error(f"Sanitization error: {str(e)}")
        return False, "Data validation error", {}
