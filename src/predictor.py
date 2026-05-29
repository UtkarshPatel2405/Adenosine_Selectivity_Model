import json
import pickle
import pandas as pd
import numpy as np
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from rdkit import Chem
from src.features import build_features, _descriptors

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

@lru_cache(maxsize=1)
def _load_db_lookup():
    p = Path("data/processed/db_lookup.json")
    return json.load(open(p, "r")) if p.exists() else {}

@lru_cache(maxsize=4)
def _load_models(mode: str = "precise"):
    if mode == "pcm":
        path = Path("models/pcm/xgboost_pcm_model.pkl")
        if not path.exists():
            raise FileNotFoundError(f"Unified PCM model NOT found at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
            
    models = {}
    model_dir = Path("models") / mode
    
    if not model_dir.exists():
        model_dir = Path("models/precise")
    if not model_dir.exists():
        model_dir = Path("models")

    for st in SUBTYPES:
        filename = model_dir / f"xgboost_{mode}_{st.lower()}_model.pkl"
        if not filename.exists():
            filename = model_dir / f"xgboost_{st.lower()}_model.pkl"
        if not filename.exists():
            filename = Path("models/precise") / f"xgboost_precise_{st.lower()}_model.pkl"
        if not filename.exists():
            filename = Path("models") / f"xgboost_{st.lower()}_model.pkl"
        
        if not filename.exists():
            raise FileNotFoundError(f"Model file NOT found for {st} ({mode} mode)")
            
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
        std_equiv = float(upper - lower) / 3.29 # Std equivalent for 90% interval
        return float(y_pred[0]), std_equiv, lower, upper
        
    elif isinstance(model_ens, (list, tuple)):
        preds = np.array([float(m.predict(x)[0]) for m in model_ens])
        mean = float(preds.mean())
        std = float(preds.std(ddof=0))
        return mean, std, mean - 1.96 * std, mean + 1.96 * std
    else:
        pred = float(model_ens.predict(x)[0])
        return pred, 0.0, pred, pred


def predict(smiles: str, threshold: float = 6.0, mode: str = "precise") -> Dict[str, Any]:
    # Handle backward compatibility mapping for modes
    if mode in ["standard", "strict"]:
        mode = "precise"
        
    scaler = _load_scaler(mode)
    lookup = _load_db_lookup()
    models = _load_models(mode=mode)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        raise ValueError("Invalid SMILES")
    canon = Chem.MolToSmiles(mol, canonical=True)

    # 1. Extract physical descriptors for metadata display
    d_vals = _descriptors(canon)
    desc_results = {
        "MW": round(float(d_vals[0]), 2), "LogP": round(float(d_vals[1]), 2),
        "HBD": int(d_vals[2]), "HBA": int(d_vals[3]),
        "RotBonds": int(d_vals[4]), "AromRings": int(d_vals[5]), "TPSA": round(float(d_vals[6]), 2)
    }

    preds, unc, intervals = {}, {}, {}
    in_db = canon in lookup

    # 2. Bioactivity Predictions
    if in_db:
        exp = lookup[canon]
        for st in SUBTYPES:
            val = exp.get(st)
            if pd.notna(val) and str(val).lower() != 'nan':
                p_val = float(val)
                preds[st], unc[st] = p_val, 0.0
                intervals[st] = {"lower": p_val, "upper": p_val, "width": 0.0}
            else:
                preds[st], unc[st] = 0.0, 0.0
                intervals[st] = {"lower": 0.0, "upper": 0.0, "width": 0.0}
        source = "database"
    else:
        x = build_features(canon, scaler)
        if mode == "pcm":
            idx_map = {st: i for i, st in enumerate(SUBTYPES)}
            for st in SUBTYPES:
                one_hot = np.zeros((len(SUBTYPES),), dtype=np.float32)
                one_hot[idx_map[st]] = 1.0
                x_pcm = np.hstack([x, one_hot]).reshape(1, -1)
                
                m, s, low, high = _ensemble_predict(models, x_pcm)
                preds[st], unc[st] = m, s
                intervals[st] = {
                    "lower": round(low, 3), 
                    "upper": round(high, 3), 
                    "width": round(high - low, 3)
                }
        else:
            for st in SUBTYPES:
                m, s, low, high = _ensemble_predict(models[st], x)
                preds[st], unc[st] = m, s
                intervals[st] = {
                    "lower": round(low, 3), 
                    "upper": round(high, 3), 
                    "width": round(high - low, 3)
                }
        source = "model"

    # 3. Direct Selectivity Predictions
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

    return {
        "smiles": canon, 
        "in_database": in_db, 
        "predictions": preds,
        "descriptors": desc_results, 
        "uncertainty": unc,
        "intervals": intervals,
        "selectivity_profile": selectivity,
        "best_target": max(preds, key=preds.get),
        "target_hits": [st for st, v in preds.items() if v >= threshold],
        "source": source
    }