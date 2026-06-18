import json
import pickle
import pandas as pd
import numpy as np
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple
from rdkit import Chem
from src.features import build_features, _descriptors


# Lazy GNN import
def _try_gnn_predict(smiles, subtype):
    try:
        from src.gnn_model import predict_gnn

        return predict_gnn(smiles, subtype)
    except (ImportError, Exception):
        return None


SUBTYPES = ["A1", "A2A", "A2B", "A3"]


@lru_cache(maxsize=4)
def _load_scaler(mode: str = "precise"):
    if mode == "antagonist_ki":
        path = Path("models/antagonist_ki/scaler_antagonist_ki.pkl")
    elif mode == "antagonist_ic50":
        path = Path("models/antagonist_ic50/scaler_antagonist_ic50.pkl")
    elif mode == "pcm":
        path = Path("models/pcm/scaler_pcm.pkl")
    else:
        path = Path("models/precise/scaler_precise.pkl")
        if not path.exists():
            path = Path("models/scaler.pkl")

    if not path.exists():
        raise FileNotFoundError(f"Scaler pipeline NOT found at {path} for mode {mode}")

    with open(path, "rb") as f:
        return pickle.load(f)


def _load_db_lookup():
    p = Path("data/processed/db_lookup_train.json")
    return json.load(open(p, "r")) if p.exists() else {}


@lru_cache(maxsize=1)
def _load_xgb_models():
    models = {}
    for st in SUBTYPES:
        filename = Path("models/precise") / f"xgboost_{st}_production.pkl"
        if not filename.exists():
            filename = Path("models") / f"xgboost_{st}_production.pkl"
        if filename.exists():
            with open(filename, "rb") as f:
                models[st] = pickle.load(f)
    return models


@lru_cache(maxsize=1)
def _load_rf_models():
    models = {}
    for st in SUBTYPES:
        filename = Path("models/precise") / f"rf_{st}_production.pkl"
        if not filename.exists():
            filename = Path("models") / f"rf_{st}_production.pkl"
        if filename.exists():
            with open(filename, "rb") as f:
                models[st] = pickle.load(f)
    return models


def _ensemble_predict(model_ens, x: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Returns (pred_mean, uncertainty_std_equiv, lower_bound, upper_bound)
    Supports legacy ensemble list, single regressor, and MapieRegressor conformal models.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Check if the model is a MAPIE conformal wrapper using duck typing
    if type(model_ens).__name__ == "CrossConformalRegressor":
        y_pred, y_pis = model_ens.predict_interval(x)
        if y_pis.ndim == 3:
            lower = float(y_pis[0, 0, 0])
            upper = float(y_pis[0, 1, 0])
        else:
            lower = float(y_pis[0, 0])
            upper = float(y_pis[0, 1])
        std_equiv = float(upper - lower) / 3.29
        return float(y_pred[0]), std_equiv, lower, upper
    elif type(model_ens).__name__ == "MapieRegressor":
        y_pred, y_pis = model_ens.predict(x, alpha=0.10)
        if y_pis.ndim == 3:
            lower = float(y_pis[0, 0, 0])
            upper = float(y_pis[0, 1, 0])
        else:
            lower = float(y_pis[0, 0])
            upper = float(y_pis[0, 1])
        std_equiv = float(upper - lower) / 3.29  # Std equivalent for 90% interval
        return float(y_pred[0]), std_equiv, lower, upper

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
    rf_models = _load_rf_models()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    canon = Chem.MolToSmiles(mol, canonical=True)

    # 1. Extract physical descriptors for metadata display
    d_vals = _descriptors(canon)
    desc_results = {
        "MW": round(float(d_vals[0]), 2),
        "LogP": round(float(d_vals[1]), 2),
        "HBD": int(d_vals[2]),
        "HBA": int(d_vals[3]),
        "RotBonds": int(d_vals[4]),
        "AromRings": int(d_vals[5]),
        "TPSA": round(float(d_vals[6]), 2),
    }

    in_db = canon in lookup

    preds = {"XGBoost": {}, "RandomForest": {}, "PyTorch": {}}
    unc = {"XGBoost": {}, "RandomForest": {}, "PyTorch": {}}
    intervals = {"XGBoost": {}, "RandomForest": {}, "PyTorch": {}}

    # Pre-computed docking scores if available
    docking_scores = None
    if in_db and isinstance(lookup[canon], dict) and "docking" in lookup[canon]:
        docking_scores = lookup[canon]["docking"]

    # Calculate features once
    x = build_features(canon, scaler) if scaler is not None else None

    # 2. Bioactivity Predictions
    source = "database" if in_db else "model"

    for st in SUBTYPES:
        if in_db:
            val = lookup[canon].get(st)
            if pd.notna(val) and str(val).lower() != "nan":
                p_val = float(val)
                for model_name in preds:
                    preds[model_name][st] = p_val
                    unc[model_name][st] = 0.0
                    intervals[model_name][st] = {
                        "lower": p_val,
                        "upper": p_val,
                        "width": 0.0,
                    }
            else:
                for model_name in preds:
                    preds[model_name][st] = 0.0
                    unc[model_name][st] = 0.0
                    intervals[model_name][st] = {
                        "lower": 0.0,
                        "upper": 0.0,
                        "width": 0.0,
                    }
        else:
            # XGBoost
            if st in xgb_models and x is not None:
                m, s, low, high = _ensemble_predict(xgb_models[st], x)
                preds["XGBoost"][st] = m
                unc["XGBoost"][st] = s
                intervals["XGBoost"][st] = {
                    "lower": round(low, 3),
                    "upper": round(high, 3),
                    "width": round(high - low, 3),
                }
            else:
                preds["XGBoost"][st], unc["XGBoost"][st], intervals["XGBoost"][st] = (
                    0.0,
                    0.0,
                    {"lower": 0.0, "upper": 0.0, "width": 0.0},
                )

            # Random Forest
            if st in rf_models and x is not None:
                m, s, low, high = _ensemble_predict(rf_models[st], x)
                preds["RandomForest"][st] = m
                unc["RandomForest"][st] = s
                intervals["RandomForest"][st] = {
                    "lower": round(low, 3),
                    "upper": round(high, 3),
                    "width": round(high - low, 3),
                }
            else:
                (
                    preds["RandomForest"][st],
                    unc["RandomForest"][st],
                    intervals["RandomForest"][st],
                ) = 0.0, 0.0, {"lower": 0.0, "upper": 0.0, "width": 0.0}

            # PyTorch (GNN)
            pred_val = _try_gnn_predict(canon, st)
            if pred_val is not None:
                m = float(pred_val)
                preds["PyTorch"][st] = m
                unc["PyTorch"][st] = 0.0
                intervals["PyTorch"][st] = {"lower": m, "upper": m, "width": 0.0}
            else:
                preds["PyTorch"][st], unc["PyTorch"][st], intervals["PyTorch"][st] = (
                    0.0,
                    0.0,
                    {"lower": 0.0, "upper": 0.0, "width": 0.0},
                )

    # 3. Direct Selectivity Predictions (using XGBoost predictions for best_target logic)
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

    # Compute best target and target hits based on XGBoost (as primary) or any other logic
    # We will use XGBoost as the reference for best target
    xgb_preds = preds["XGBoost"]

    # If XGBoost prediction is available, use it. Otherwise fallback to PyTorch
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
        "target_hits": [st for st, v in ref_preds.items() if v >= threshold]
        if ref_preds
        else [],
        "source": source,
        "docking_scores": docking_scores,
    }
