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
    except (ImportError, Exception):
        return None


@lru_cache(maxsize=4)
def _load_scaler(mode: str = "precise"):
    candidate_paths = [
        MODELS_DIR / mode / f"scaler_{mode}.pkl",
        MODELS_DIR / "scaler.pkl",
    ]
    for path in candidate_paths:
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"No scaler found for mode={mode} in any path.")


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
        if path.exists():
            with open(path, "rb") as f:
                models[st] = pickle.load(f)
        else:
            logger.warning("XGBoost model for %s not found at %s", st, path)
    return models


@lru_cache(maxsize=1)
def _load_rf_models():
    models = {}
    for st in SUBTYPES:
        path = MODELS_DIR / "precise" / f"rf_{st}_production.pkl"
        if not path.exists():
            path = MODELS_DIR / f"rf_{st}_production.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[st] = pickle.load(f)
    return models


def _ensemble_predict(model_ens, x: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Returns (pred_mean, uncertainty_std_equiv, lower_bound, upper_bound).

    Production models are raw XGBRegressor (no MAPIE wrapper in pickle).

    Falls back gracefully for legacy models (CrossConformalRegressor, MapieRegressor,
    list ensembles, raw sklearn, etc.).
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    model_type_name = type(model_ens).__name__

    if model_type_name == "CrossConformalRegressor":
        try:
            y_pred, y_pis = model_ens.predict_interval(x)
            if y_pis.ndim == 3:
                lower = float(y_pis[0, 0, 0])
                upper = float(y_pis[0, 1, 0])
            else:
                lower = float(y_pis[0, 0])
                upper = float(y_pis[0, 1])
            std_equiv = float(upper - lower) / 3.29
            logger.debug("Conformal pred: %.3f [%.3f, %.3f] (std=%.3f)",
                         float(y_pred[0]), lower, upper, std_equiv)
            return float(y_pred[0]), std_equiv, lower, upper
        except Exception as e:
            logger.warning("CrossConformalRegressor predict_interval failed: %s. Falling back.", e)
            pred = float(model_ens.predict(x)[0])
            return pred, 0.0, pred, pred

    elif model_type_name == "MapieRegressor":
        try:
            y_pred, y_pis = model_ens.predict(x)
            if y_pis.ndim == 3:
                lower = float(y_pis[0, 0, 0])
                upper = float(y_pis[0, 1, 0])
            else:
                lower = float(y_pis[0, 0])
                upper = float(y_pis[0, 1])
            std_equiv = float(upper - lower) / 3.29
            return float(y_pred[0]), std_equiv, lower, upper
        except Exception as e:
            logger.warning("MapieRegressor predict failed: %s. Falling back.", e)
            pred = float(model_ens.predict(x)[0])
            return pred, 0.0, pred, pred

    elif isinstance(model_ens, (list, tuple)):
        preds = np.array([float(m.predict(x)[0]) for m in model_ens])
        mean = float(preds.mean())
        std = float(preds.std(ddof=0))
        return mean, std, mean - 1.96 * std, mean + 1.96 * std

    else:
        pred = float(model_ens.predict(x)[0])
        return pred, 0.0, pred, pred


def predict(smiles: str, threshold: float = 6.0) -> Dict[str, Any]:
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

    docking_scores = None
    if in_db and isinstance(lookup[canon], dict) and "docking" in lookup[canon]:
        docking_scores = lookup[canon]["docking"]

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
            _rf = _load_rf_models()
            if st in _rf and x is not None:
                m, s, low, high = _ensemble_predict(_rf[st], x)
                preds["RandomForest"][st] = m
                unc["RandomForest"][st] = s
                intervals["RandomForest"][st] = {"lower": round(low, 3), "upper": round(high, 3), "width": round(high - low, 3)}
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
        "docking_scores": docking_scores,
    }
