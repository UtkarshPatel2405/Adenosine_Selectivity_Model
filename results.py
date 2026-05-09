import json
import pickle
import os
from datetime import datetime
from pathlib import Path
import numpy as np

# Pipeline components
from src.data_loader import load_and_clean
try:
    from src.data_splitter import scaffold_split
except Exception:
    from src.scaffold_split import scaffold_split

from src.features import build_features
from src.predictor import SUBTYPES, _load_scaler

def _write_json(path: str, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def local_predict_for_report(smiles: str, models: dict, lookup: dict, scaler, threshold: float):
    
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return {"error": "Invalid SMILES"}
    canon = Chem.MolToSmiles(mol, canonical=True)

    if canon in lookup:
        preds = {st: float(lookup[canon].get(st, 0.0)) for st in SUBTYPES}
        source = "database"
    else:
        x = build_features(canon, scaler)
        preds = {st: float(models[st].predict(x.reshape(1, -1))[0]) for st in SUBTYPES}
        source = "model"

    return {
        "smiles": canon,
        "predictions": preds,
        "best_target": max(preds, key=preds.get),
        "target_hits": [st for st, v in preds.items() if v >= threshold],
        "source": source
    }

def run_mode(mode: str, out_dir: str, hit_threshold: float = 6.0):
    print(f"\n>>> Running Pipeline: {mode.upper()}")
    
   
    df, lookup = load_and_clean("data/raw/AR_all_unique_parents_with_smiles.csv", mode=mode)
    scaler = _load_scaler()
    
   
    models = {}
    for st in SUBTYPES:
        file_name = f"xgboost_{st.lower()}_model.pkl"
        
       
        primary = Path("models") / mode / file_name
        
        fallback = Path("models") / "standard" / file_name
        
        load_path = primary if primary.exists() else fallback
        
        if not load_path.exists():
            print(f"CRITICAL ERROR: {file_name} missing from both {mode} and standard folders.")
            continue
            
        with open(load_path, "rb") as f:
            models[st] = pickle.load(f)

    
    db_smiles = df["canonical_smiles"].dropna().unique().tolist()[:5]
    db_results = [
        {"smiles": s, "result": local_predict_for_report(s, models, lookup, scaler, hit_threshold)} 
        for s in db_smiles
    ]

    novel_smiles = ["c1ccc2[nH]ccc2c1", "CCN(CC)CCc1ccc2[nH]ccc2c1", "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"]
    novel_results = []
    for s in novel_smiles:
        res = local_predict_for_report(s, models, lookup, scaler, hit_threshold)
        novel_results.append({"smiles": s, "result": res})

    
    _write_json(f"{out_dir}/run_summary.json", {
        "mode": mode,
        "n_rows_clean": len(df),
        "n_lookup_smiles": len(lookup),
        "timestamp": datetime.now().isoformat()
    })
    _write_json(f"{out_dir}/predictor_db_examples.json", db_results)
    _write_json(f"{out_dir}/predictor_novel_examples.json", novel_results)
    print(f"DONE: Results saved to {out_dir}")

def main():
   
    for m in ["standard", "strict"]:
        os.makedirs(f"outputs/validoutput/{m}", exist_ok=True)
        os.makedirs(f"models/{m}", exist_ok=True)

   
    for mode in ["standard", "strict"]:
        run_mode(mode, f"outputs/validoutput/{mode}")

if __name__ == "__main__":
    main()