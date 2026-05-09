from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def _load_json(path: str) -> dict:
    """Safely load JSON, returning empty dict if file is missing."""
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_evaluation_tables(base: str = "outputs/validoutput/standard") -> Tuple[pd.DataFrame, pd.DataFrame]:
    report = _load_json(f"{base}/evaluation_report.json")
    if not report:
        return pd.DataFrame(), pd.DataFrame()

    overall = report.get("overall", {})
    overall_df = pd.DataFrame([{
        "MAE": round(overall.get("model_mae", 0), 4),
        "RMSE": round(overall.get("model_rmse", 0), 4),
        "R2": round(overall.get("model_r2", 0), 4),
        "Baseline R2": round(overall.get("baseline_r2", 0), 4),
    }])

    per = report.get("per_subtype", {})
    rows = []
    for st, m in per.items():
        if m.get("skipped"): continue
        rows.append({
            "Subtype": st,
            "MAE": round(m.get("model_mae", 0), 4),
            "RMSE": round(m.get("model_rmse", 0), 4),
            "R2": round(m.get("model_r2", 0), 4),
            "n_test": m.get("n_test"),
        })
    per_df = pd.DataFrame(rows).sort_values("Subtype") if rows else pd.DataFrame()
    return overall_df, per_df

def load_fingerprint_comparison(base: str = "outputs/validoutput/standard") -> pd.DataFrame:
    path = Path(f"{base}/fingerprint_comparison.csv")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def load_scaffold_ood(base: str = "outputs/validoutput/standard") -> pd.DataFrame:
    data = _load_json(f"{base}/scaffold_ood_report.json")
    ood = data.get("ood_by_scaffold", {})
    if not ood:
        return pd.DataFrame()
    rows = [{"Scaffold": k, **v} for k, v in ood.items()]
    return pd.DataFrame(rows).sort_values("count", ascending=False)

def load_run_summary(base: str = "outputs/validoutput/standard") -> pd.DataFrame:
    s = _load_json(f"{base}/run_summary.json")
    if not s:
        return pd.DataFrame([{"Status": "Data not found"}])
    return pd.DataFrame([{
        "Mode": s.get("mode", "N/A").upper(),
        "Total Rows": s.get("n_rows_clean"),
        "DB Size": s.get("n_lookup_smiles"),
        "Train": s.get("split", {}).get("n_train"),
        "Test": s.get("split", {}).get("n_test"),
        "X_train_shape": str(s.get("features", {}).get("X_train_shape")),
    }])

def _examples_to_df(path: str) -> pd.DataFrame:
    """Formats prediction examples with Reliability labels for the UI."""
    from src.chem_utils import nearest_tanimoto
    data = _load_json(path)
    if not data: return pd.DataFrame()
    
    rows = []
    for item in data:
        smi = item.get("smiles")
        r = item.get("result", {})
        preds = r.get("predictions", {})
        preds_str = ", ".join([f"{k}: {v:.2f}" for k, v in preds.items()]) if preds else "N/A"
        
        try:
            sim = nearest_tanimoto(smi)
            if sim is not None:
                if sim >= 0.6: label = "High"
                elif sim >= 0.4: label = "Med"
                else: label = "Low"
                sim_str = f"{sim:.3f} ({label})"
            else:
                sim_str = "0"
        except Exception:
            sim_str = "0"

        rows.append({
            "SMILES": smi,
            "Source": r.get("source"),
            "Best": r.get("best_target"),
            "Predictions": preds_str,
            "Similarity (AD)": sim_str,
            "Hits": ", ".join(r.get("target_hits", [])) if r.get("target_hits") else "None",
        })
    return pd.DataFrame(rows)

def load_mode_examples(mode: str, base: str = "outputs/validoutput"):
    base_dir = f"{base}/{mode}"
    db_df = _examples_to_df(f"{base_dir}/predictor_db_examples.json")
    novel_df = _examples_to_df(f"{base_dir}/predictor_novel_examples.json")
    summary_df = load_run_summary(base=base_dir)
    return summary_df, db_df, novel_df

def load_examples(base: str = "outputs/validoutput/standard") -> Tuple[pd.DataFrame, pd.DataFrame]:
    db_df = _examples_to_df(f"{base}/predictor_db_examples.json")
    novel_df = _examples_to_df(f"{base}/predictor_novel_examples.json")
    return db_df, novel_df

def outputs_exist(base: str = "outputs/validoutput/standard") -> Dict[str, bool]:
    files = [
        "run_summary.json",
        "predictor_db_examples.json",
        "predictor_novel_examples.json",
        "evaluation_report.json",
    ]
    return {f: Path(f"{base}/{f}").exists() for f in files}