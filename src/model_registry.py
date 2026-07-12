"""
Model versioning and tracking system for MLOps.
Manages model artifacts, metadata, and version history.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

MODEL_DIR = Path("models")
REGISTRY_FILE = MODEL_DIR / "model_registry.json"


def get_model_version() -> str:
    """Generate version string from timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def register_model(
    model_path: str,
    model_name: str,
    version: Optional[str] = None,
    metrics: Optional[Dict] = None,
    tags: Optional[list] = None,
) -> Dict:
    """Register a new model version in the registry."""
    if version is None:
        version = get_model_version()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load or initialize registry
    registry = {}
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            registry = json.load(f)
    
    model_entry = {
        "version": version,
        "model_name": model_name,
        "model_path": str(model_path),
        "registered_at": datetime.now().isoformat(),
        "metrics": metrics or {},
        "tags": tags or [],
        "is_production": False,
    }
    
    if model_name not in registry:
        registry[model_name] = []
    
    registry[model_name].append(model_entry)
    
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"[model_registry] Registered {model_name}:{version}")
    return model_entry


def promote_model(model_name: str, version: str) -> bool:
    """Promote a model version to production."""
    if not REGISTRY_FILE.exists():
        return False
    
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    
    if model_name not in registry:
        return False
    
    # Demote previous production model
    for entry in registry[model_name]:
        if entry["is_production"]:
            entry["is_production"] = False
            print(f"[model_registry] Demoted {model_name}:{entry['version']}")
    
    # Promote new version
    for entry in registry[model_name]:
        if entry["version"] == version:
            entry["is_production"] = True
            entry["promoted_at"] = datetime.now().isoformat()
            print(f"[model_registry] Promoted {model_name}:{version} to production")
            break
    
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
    
    return True


def get_production_model(model_name: str) -> Optional[Dict]:
    """Get the current production model."""
    if not REGISTRY_FILE.exists():
        return None
    
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    
    if model_name not in registry:
        return None
    
    for entry in registry[model_name]:
        if entry["is_production"]:
            return entry
    
    return None


def list_model_versions(model_name: str) -> list:
    """List all versions of a model."""
    if not REGISTRY_FILE.exists():
        return []
    
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    
    return registry.get(model_name, [])
