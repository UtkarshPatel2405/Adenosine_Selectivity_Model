import logging
import os
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

logger = logging.getLogger(__name__)

_MLFLOW_AVAILABLE = False
try:
    import mlflow
    _MLFLOW_AVAILABLE = bool(MLFLOW_TRACKING_URI)
    if _MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info("MLflow tracking enabled: uri=%s, experiment=%s",
                     MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)
    else:
        logger.info("MLflow tracking disabled (MLFLOW_TRACKING_URI not set).")
except ImportError:
    logger.info("MLflow not installed. Experiment tracking disabled.")
except Exception as e:
    logger.warning("MLflow initialization failed: %s. Tracking disabled.", e)


def mlflow_run(run_name: Optional[str] = None):
    """
    Decorator that wraps a function in an MLflow run for experiment tracking.
    If MLflow is unavailable, the function runs normally with no-op.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _MLFLOW_AVAILABLE:
                return func(*args, **kwargs)
            with mlflow.start_run(run_name=run_name or func.__name__):
                mlflow.set_tag("function", func.__name__)
                result = func(*args, **kwargs)
                if isinstance(result, dict):
                    mlflow.log_metrics({
                        k: v for k, v in _flatten_dict(result).items()
                        if isinstance(v, (int, float))
                    })
                return result
        return wrapper
    return decorator


def log_params(params: Dict[str, Any]):
    if _MLFLOW_AVAILABLE:
        mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    if _MLFLOW_AVAILABLE:
        mlflow.log_metrics(metrics, step=step)


def log_artifact(local_path: str):
    if _MLFLOW_AVAILABLE:
        mlflow.log_artifact(local_path)


def log_model(model, artifact_path: str, **kwargs):
    if _MLFLOW_AVAILABLE:
        try:
            mlflow.sklearn.log_model(model, artifact_path, **kwargs)
        except Exception:
            mlflow.pyfunc.log_model(artifact_path, **kwargs)


def set_tags(tags: Dict[str, str]):
    if _MLFLOW_AVAILABLE:
        mlflow.set_tags(tags)


def _flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, float]:
    """Flatten nested dicts for MLflow metric logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}_{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key).items())
        elif isinstance(v, (int, float)):
            items.append((new_key, v))
    return dict(items)
