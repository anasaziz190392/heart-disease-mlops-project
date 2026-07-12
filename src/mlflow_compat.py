"""
mlflow_compat.py
-----------------
Thin compatibility layer so the training pipeline always works:

  * If the real `mlflow` package is installed (as required by
    requirements.txt in any normal / CI / Docker environment), it is used
    directly and this module simply re-exports it.
  * If `mlflow` is NOT installed (e.g. an offline sandbox with no network
    access to PyPI), a minimal drop-in shim is provided that implements the
    same call pattern (`start_run`, `log_param`, `log_params`, `log_metric`,
    `log_metrics`, `log_artifact`, `set_experiment`, `sklearn.log_model`)
    and writes everything to a local `mlruns_local/` folder in a structure
    that mirrors MLflow's own run layout (params/, metrics/, artifacts/).

This guarantees `python src/train.py` is always runnable end-to-end for
grading/demo purposes, while the *real* MLflow UI (`mlflow ui`) is what you
should use for the actual submission once `pip install -r requirements.txt`
has been run.
"""
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime

try:
    import mlflow  # noqa: F401
    import mlflow.sklearn  # noqa: F401

    MLFLOW_AVAILABLE = True

except ImportError:
    MLFLOW_AVAILABLE = False

    _BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "mlruns_local")
    _current_run = {"id": None, "dir": None, "experiment": "Default"}

    class _SklearnShim:
        @staticmethod
        def log_model(model, artifact_path, **kwargs):
            import pickle
            run_dir = _current_run["dir"]
            out_dir = os.path.join(run_dir, "artifacts", artifact_path)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
                pickle.dump(model, f)

    class _MlflowShim:
        sklearn = _SklearnShim()

        @staticmethod
        def set_experiment(name):
            _current_run["experiment"] = name

        @staticmethod
        @contextmanager
        def start_run(run_name=None):
            run_id = str(uuid.uuid4())[:8]
            run_dir = os.path.join(
                _BASE_DIR, _current_run["experiment"],
                f"{run_name or 'run'}_{run_id}",
            )
            os.makedirs(run_dir, exist_ok=True)
            _current_run["id"] = run_id
            _current_run["dir"] = run_dir
            meta = {"run_name": run_name, "run_id": run_id, "start_time": datetime.now().isoformat()}
            with open(os.path.join(run_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            try:
                yield meta
            finally:
                pass

        @staticmethod
        def log_param(key, value):
            _append_json(os.path.join(_current_run["dir"], "params.json"), key, value)

        @staticmethod
        def log_params(d):
            for k, v in d.items():
                _MlflowShim.log_param(k, v)

        @staticmethod
        def log_metric(key, value, step=None):
            _append_json(os.path.join(_current_run["dir"], "metrics.json"), key, value)

        @staticmethod
        def log_metrics(d):
            for k, v in d.items():
                _MlflowShim.log_metric(k, v)

        @staticmethod
        def log_artifact(path, artifact_path=None):
            run_dir = _current_run["dir"]
            dest_dir = os.path.join(run_dir, "artifacts", artifact_path or "")
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(path, dest_dir)

        @staticmethod
        def end_run():
            pass

    def _append_json(path, key, value):
        data = {}
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
        data[key] = value
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    mlflow = _MlflowShim()

print(f"[mlflow_compat] Using {'REAL mlflow' if MLFLOW_AVAILABLE else 'LOCAL FALLBACK shim (install mlflow for full UI: pip install mlflow)'}")
