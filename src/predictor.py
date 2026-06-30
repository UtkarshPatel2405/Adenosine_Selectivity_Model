import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

from src.features import build_features, _descriptors
from src.config import SUBTYPES, MODELS_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


def _try_gnn_predict(smiles, subtype):
    try:
        from src.gnn_model import predict_gnn
        return predict_gnn(smiles, subtype)
    except ImportError:
        return None
    except Exception as e:
        logger.warning("GNN prediction failed for %s/%s: %s", subtype, smiles[:30], e)
        return None


@lru_cache(maxsize=4)
def _load_scaler(mode: str = "precise"):
    candidate_paths = [
        MODELS_DIR / mode / f"scaler_{mode}.pkl",
        MODELS_DIR / mode / "scaler.pkl",
        MODELS_DIR / "scaler.pkl",
    ]
    last_err = None
    for path in candidate_paths:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error("Failed to load scaler from %s: %s", path, e)
                last_err = e
    raise FileNotFoundError(f"No scaler found for mode={mode} in any path. Last error: {last_err}")


@lru_cache(maxsize=1)
def _load_db_lookup():
    p = PROCESSED_DATA_DIR / "db_lookup_train.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@lru_cache(maxsize=1)
def _load_xgb_models():
    models = {}
    for st in SUBTYPES:
        path = MODELS_DIR / "precise" / f"xgboost_{st}_production.pkl"
        if not path.exists():
            path = MODELS_DIR / f"xgboost_{st}_production.pkl"
        if not path.exists():
            path = MODELS_DIR / f"xgboost_precise_{st.lower()}_model.pkl"
        if not path.exists():
            path = MODELS_DIR / f"xgboost_{st.lower()}_model.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    models[st] = pickle.load(f)
            except Exception as e:
                logger.error("Failed to load XGBoost model for %s from %s: %s", st, path, e)
        else:
            logger.warning("XGBoost model for %s not found at any path", st)
    return models


@lru_cache(maxsize=1)
def _load_rf_models():
    models = {}
    for st in SUBTYPES:
        path = MODELS_DIR / "precise" / f"rf_{st}_production.pkl"
        if not path.exists():
            path = MODELS_DIR / f"rf_{st}_production.pkl"
        if not path.exists():
            path = MODELS_DIR / f"rf_precise_{st.lower()}_model.pkl"
        if not path.exists():
            path = MODELS_DIR / f"rf_{st.lower()}_model.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    models[st] = pickle.load(f)
            except Exception as e:
                logger.error("Failed to load RandomForest model for %s from %s: %s", st, path, e)
        else:
            logger.warning("RandomForest model for %s not found at any path", st)
    return models


def _ensemble_predict(model_ens, x: np.ndarray) -> Tuple[Any, Any, Any, Any]:
    """
    Returns (pred_mean, uncertainty_std_equiv, lower_bound, upper_bound).

    Supports both single sample and batch inputs.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    is_batch = x.shape[0] > 1
    model_type_name = type(model_ens).__name__

    if model_type_name == "CrossConformalRegressor":
        try:
            y_pred, y_pis = model_ens.predict_interval(x)
            if y_pis.ndim == 3:
                lower = y_pis[:, 0, 0]
                upper = y_pis[:, 1, 0]
            else:
                lower = y_pis[:, 0]
                upper = y_pis[:, 1]
            std_equiv = (upper - lower) / 3.29
            if not is_batch:
                return float(y_pred[0]), float(std_equiv[0]), float(lower[0]), float(upper[0])
            return y_pred, std_equiv, lower, upper
        except Exception as e:
            logger.warning("CrossConformalRegressor predict_interval failed: %s. Falling back.", e)
            y_pred = model_ens.predict(x)
            if not is_batch:
                return float(y_pred[0]), 0.0, float(y_pred[0]), float(y_pred[0])
            zeros = np.zeros_like(y_pred)
            return y_pred, zeros, y_pred, y_pred

    elif model_type_name == "MapieRegressor":
        try:
            y_pred, y_pis = model_ens.predict(x)
            if y_pis.ndim == 3:
                lower = y_pis[:, 0, 0]
                upper = y_pis[:, 1, 0]
            else:
                lower = y_pis[:, 0]
                upper = y_pis[:, 1]
            std_equiv = (upper - lower) / 3.29
            if not is_batch:
                return float(y_pred[0]), float(std_equiv[0]), float(lower[0]), float(upper[0])
            return y_pred, std_equiv, lower, upper
        except Exception as e:
            logger.warning("MapieRegressor predict failed: %s. Falling back.", e)
            y_pred = model_ens.predict(x)
            if not is_batch:
                return float(y_pred[0]), 0.0, float(y_pred[0]), float(y_pred[0])
            zeros = np.zeros_like(y_pred)
            return y_pred, zeros, y_pred, y_pred

    elif isinstance(model_ens, (list, tuple)):
        member_preds = np.array([m.predict(x) for m in model_ens])
        mean = member_preds.mean(axis=0)
        std = member_preds.std(axis=0, ddof=0)
        if not is_batch:
            return float(mean[0]), float(std[0]), float(mean[0] - 1.96 * std[0]), float(mean[0] + 1.96 * std[0])
        return mean, std, mean - 1.96 * std, mean + 1.96 * std

    else:
        pred = model_ens.predict(x)
        if not is_batch:
            return float(pred[0]), 0.0, float(pred[0]), float(pred[0])
        zeros = np.zeros_like(pred)
        return pred, zeros, pred, pred


def predict(smiles: str, threshold: float = 6.0, run_rf: bool = False) -> Dict[str, Any]:
    lookup = _load_db_lookup()

    try:
        scaler = _load_scaler("precise")
    except Exception:
        scaler = None

    xgb_models = _load_xgb_models()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    canon = Chem.MolToSmiles(mol, canonical=True)

    d_vals = _descriptors(canon)
    desc_results = {
        "MW": round(float(d_vals[0]), 2), "LogP": round(float(d_vals[1]), 2),
        "HBD": int(d_vals[2]), "HBA": int(d_vals[3]),
        "RotBonds": int(d_vals[4]), "AromRings": int(d_vals[5]), "TPSA": round(float(d_vals[6]), 2),
    }

    in_db = canon in lookup

    preds = {"XGBoost": {}, "RandomForest": {}, "PyTorch": {}}
    unc = {"XGBoost": {}, "RandomForest": {}, "PyTorch": {}}
    intervals = {"XGBoost": {}, "RandomForest": {}, "PyTorch": {}}


    x = build_features(canon, scaler) if scaler is not None else None

    source = "database" if in_db else "model"

    for st in SUBTYPES:
        if in_db:
            val = lookup[canon].get(st)
            if pd.notna(val) and str(val).lower() != 'nan':
                p_val = float(val)
                for model_name in preds:
                    preds[model_name][st] = p_val
                    unc[model_name][st] = 0.0
                    intervals[model_name][st] = {"lower": p_val, "upper": p_val, "width": 0.0}
            else:
                for model_name in preds:
                    preds[model_name][st] = 0.0
                    unc[model_name][st] = 0.0
                    intervals[model_name][st] = {"lower": 0.0, "upper": 0.0, "width": 0.0}
        else:
            if st in xgb_models and x is not None:
                m, s, low, high = _ensemble_predict(xgb_models[st], x)
                preds["XGBoost"][st] = m
                unc["XGBoost"][st] = s
                intervals["XGBoost"][st] = {"lower": round(low, 3), "upper": round(high, 3), "width": round(high - low, 3)}
            else:
                preds["XGBoost"][st], unc["XGBoost"][st], intervals["XGBoost"][st] = 0.0, 0.0, {"lower": 0.0, "upper": 0.0, "width": 0.0}

            # Lazy-load RF on demand (avoids 105MB deserialization on cold start)
            if run_rf:
                _rf = _load_rf_models()
                if st in _rf and x is not None:
                    m, s, low, high = _ensemble_predict(_rf[st], x)
                    preds["RandomForest"][st] = m
                    unc["RandomForest"][st] = s
                    intervals["RandomForest"][st] = {"lower": round(low, 3), "upper": round(high, 3), "width": round(high - low, 3)}
                else:
                    preds["RandomForest"][st], unc["RandomForest"][st], intervals["RandomForest"][st] = 0.0, 0.0, {"lower": 0.0, "upper": 0.0, "width": 0.0}
            else:
                preds["RandomForest"][st], unc["RandomForest"][st], intervals["RandomForest"][st] = 0.0, 0.0, {"lower": 0.0, "upper": 0.0, "width": 0.0}

            pred_val = _try_gnn_predict(canon, st)
            if pred_val is not None:
                m = float(pred_val)
                preds["PyTorch"][st] = m
                unc["PyTorch"][st] = 0.0
                intervals["PyTorch"][st] = {"lower": m, "upper": m, "width": 0.0}
            else:
                preds["PyTorch"][st], unc["PyTorch"][st], intervals["PyTorch"][st] = 0.0, 0.0, {"lower": 0.0, "upper": 0.0, "width": 0.0}

    selectivity = {}
    pairs = [("A2A", "A1"), ("A2A", "A3")]
    try:
        from src.selectivity_models import predict_direct_selectivity
        for subA, subB in pairs:
            pred_sel = predict_direct_selectivity(canon, subA, subB)
            if pred_sel is not None:
                selectivity[f"{subA}_vs_{subB}"] = round(pred_sel, 3)
    except Exception:
        pass

    xgb_preds = preds["XGBoost"]
    ref_preds = xgb_preds if sum(xgb_preds.values()) > 0 else preds["PyTorch"]

    return {
        "smiles": canon,
        "in_database": in_db,
        "predictions": preds,
        "descriptors": desc_results,
        "uncertainty": unc,
        "intervals": intervals,
        "selectivity_profile": selectivity,
        "best_target": max(ref_preds, key=ref_preds.get) if ref_preds else None,
        "target_hits": [st for st, v in ref_preds.items() if v >= threshold] if ref_preds else [],
        "source": source,
    }
