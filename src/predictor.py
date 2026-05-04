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

@lru_cache(maxsize=1)
def _load_scaler():
    with open("models/scaler.pkl", "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def _load_db_lookup():
    p = Path("data/processed/db_lookup.json")
    return json.load(open(p, "r")) if p.exists() else {}

@lru_cache(maxsize=1)
def _load_models():
    models = {}
    for st in SUBTYPES:
        
        filename = f"models/xgboost_{st.lower()}_model.pkl" 
        with open(filename, "rb") as f:
            models[st] = pickle.load(f)
    return models
def _ensemble_predict(model_ens, x: np.ndarray) -> Tuple[float, float]:
    
    if isinstance(model_ens, (list, tuple)):
        preds = np.array([float(m.predict(x.reshape(1, -1))[0]) for m in model_ens])
        return float(preds.mean()), float(preds.std(ddof=0))
    else:
        
        pred = float(model_ens.predict(x.reshape(1, -1))[0])
        return pred, 0.0

def predict(smiles: str, threshold: float = 6.0) -> Dict[str, Any]:
    scaler = _load_scaler()
    lookup = _load_db_lookup()
    models = _load_models()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError("Invalid SMILES")
    canon = Chem.MolToSmiles(mol, canonical=True)

    # 1. Always extract the 7 descriptors[cite: 10, 19]
    d_vals = _descriptors(canon)
    desc_results = {
        "MW": round(float(d_vals[0]), 2), "LogP": round(float(d_vals[1]), 2),
        "HBD": int(d_vals[2]), "HBA": int(d_vals[3]),
        "RotBonds": int(d_vals[4]), "AromRings": int(d_vals[5]), "TPSA": round(float(d_vals[6]), 2)
    }

    preds, unc = {}, {}
    in_db = canon in lookup

    # 2. Professor's Hierarchy[cite: 12, 19]
    if in_db:
        exp = lookup[canon]
        for st in SUBTYPES:
            val = exp.get(st)
            if pd.notna(val) and str(val).lower() != 'nan':
                preds[st], unc[st] = float(val), 0.0
            else:
                preds[st], unc[st] = 0.0, 0.0 # Assumed zero
        source = "database"
    else:
        x = build_features(canon, scaler)
        for st in SUBTYPES:
            m, s = _ensemble_predict(models[st], x)
            preds[st], unc[st] = m, s
        source = "model"

    return {
        "smiles": canon, "in_database": in_db, "predictions": preds,
        "descriptors": desc_results, "uncertainty": unc,
        "best_target": max(preds, key=preds.get),
        "target_hits": [st for st, v in preds.items() if v >= threshold],
        "source": source
    }