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

@lru_cache(maxsize=2)
def _load_models(mode: str = "standard"):
    models = {}
    # 1. Try the mode-specific subfolder (standard/strict)
    model_dir = Path("models") / mode
    
    # 2. Fallback: If the subfolder doesn't exist, use the root models folder
    if not model_dir.exists():
        print(f"WARNING: {model_dir} not found. Falling back to root 'models/' directory.")
        model_dir = Path("models")

    for st in SUBTYPES:
        filename = model_dir / f"xgboost_{st.lower()}_model.pkl" 
        
        if not filename.exists():
            raise FileNotFoundError(f"Model file NOT found: {filename}")
            
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

def predict(smiles: str, threshold: float = 6.0, mode: str = "standard") -> Dict[str, Any]:
    scaler = _load_scaler()
    lookup = _load_db_lookup()
    models = _load_models(mode=mode)

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