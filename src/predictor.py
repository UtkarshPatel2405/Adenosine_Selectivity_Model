import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from rdkit import Chem

from src.features import build_features

SUBTYPES = ["A1", "A2A", "A2B", "A3"]


def _canonicalize(smiles: str) -> Optional[str]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_scaler():
    return _load_pickle("models/scaler.pkl")


@lru_cache(maxsize=1)
def _load_db_lookup() -> Dict[str, Dict[str, float]]:
    p = Path("data/processed/db_lookup.json")
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for st in SUBTYPES:
        models[st] = _load_pickle(f"models/xgb_{st}_ens.pkl")
    models["global"] = _load_pickle("models/xgb_global_ens.pkl")
    return models


def _ensemble_predict(model_ens, x: np.ndarray) -> Tuple[float, float]:
    preds = np.array([float(m.predict(x.reshape(1, -1))[0]) for m in model_ens], dtype=float)
    return float(preds.mean()), float(preds.std(ddof=0))


def predict(smiles: str, threshold: float = 6.0) -> Dict[str, Any]:
    scaler = _load_scaler()
    lookup = _load_db_lookup()
    models = _load_models()

    canon = _canonicalize(smiles)
    if canon is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = build_features(canon, scaler)

    if canon in lookup:
        exp = lookup[canon]
        preds: Dict[str, Optional[float]] = {st: (float(exp[st]) if st in exp else None) for st in SUBTYPES}

        available = sorted([v for v in preds.values() if v is not None], reverse=True)
        if available:
            best_target = max(preds, key=lambda k: preds[k] if preds[k] is not None else -1e9)
        else:
            best_target = "A2A"  

        selectivity = float(available[0] - available[1]) if len(available) >= 2 else None
        hits = [st for st, v in preds.items() if v is not None and v > threshold]

        return {
            "in_database": True,
            "predictions": preds,
            "best_target": best_target,
            "target_hits": hits,
            "selectivity_score": selectivity,
            "uncertainty": {st: 0.0 for st in SUBTYPES},
            "pains_alert": False,
            "toxicophore_alert": False,
            "source": "database",
        }

    preds2: Dict[str, float] = {}
    unc: Dict[str, float] = {}

    for st in SUBTYPES:
        mean_pred, std_pred = _ensemble_predict(models[st], x)
        preds2[st] = mean_pred
        unc[st] = std_pred

    best_target = max(preds2, key=lambda k: preds2[k])
    sorted_vals = sorted(preds2.values(), reverse=True)
    selectivity = float(sorted_vals[0] - sorted_vals[1])
    hits = [st for st, v in preds2.items() if v > threshold]

    return {
        "in_database": False,
        "predictions": preds2,
        "best_target": best_target,
        "target_hits": hits,
        "selectivity_score": selectivity,
        "uncertainty": unc,
        "pains_alert": False,
        "toxicophore_alert": False,
        "source": "model",
    }
