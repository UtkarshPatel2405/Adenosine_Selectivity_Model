from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_evaluation_tables(base: str = "outputs") -> Tuple[pd.DataFrame, pd.DataFrame]:
    report = _load_json(f"{base}/evaluation_report.json")
    overall = report.get("overall", {})
    overall_df = pd.DataFrame(
        [{
            "model_mae": overall.get("model_mae"),
            "model_rmse": overall.get("model_rmse"),
            "model_r2": overall.get("model_r2"),
            "baseline_mae": overall.get("baseline_mae"),
            "baseline_rmse": overall.get("baseline_rmse"),
            "baseline_r2": overall.get("baseline_r2"),
        }]
    )

    per = report.get("per_subtype", {})
    rows = []
    for st, m in per.items():
        if m.get("skipped"):
            continue
        rows.append({
            "Subtype": st,
            "MAE": m.get("model_mae"),
            "RMSE": m.get("model_rmse"),
            "R2": m.get("model_r2"),
            "Baseline MAE": m.get("baseline_mae"),
            "Baseline RMSE": m.get("baseline_rmse"),
            "Baseline R2": m.get("baseline_r2"),
            "n_test": m.get("n_test"),
        })
    per_df = pd.DataFrame(rows)
    return overall_df, per_df


def load_fingerprint_comparison(base: str = "outputs") -> pd.DataFrame:
    return pd.read_csv(f"{base}/fingerprint_comparison.csv")


def load_scaffold_ood(base: str = "outputs") -> pd.DataFrame:
    ood = _load_json(f"{base}/scaffold_ood_report.json").get("ood_by_scaffold", {})
    rows = [{"Scaffold": k, **v} for k, v in ood.items()]
    return pd.DataFrame(rows)


def load_run_summary(base: str = "outputs") -> pd.DataFrame:
    s = _load_json(f"{base}/run_summary.json")
    return pd.DataFrame([{
        "mode": s.get("mode"),
        "n_rows_clean": s.get("n_rows_clean"),
        "n_lookup_smiles": s.get("n_lookup_smiles"),
        "n_train": s.get("split", {}).get("n_train"),
        "n_test": s.get("split", {}).get("n_test"),
        "X_train_shape": s.get("features", {}).get("X_train_shape"),
        "X_test_shape": s.get("features", {}).get("X_test_shape"),
    }])
def load_mode_examples(mode: str, base: str = "outputs/validoutput"):
    base_dir = f"{base}/{mode}"
    db_df = _examples_to_df(f"{base_dir}/predictor_db_examples.json")
    novel_df = _examples_to_df(f"{base_dir}/predictor_novel_examples.json")
    summary_df = load_run_summary(base=base_dir)
    return summary_df, db_df, novel_df

def _examples_to_df(path: str) -> pd.DataFrame:
    data = _load_json(path)
    rows = []
    for item in data:
        r = item.get("result", {})
        rows.append({
            "SMILES": item.get("smiles"),
            "Source": r.get("source"),
            "Best": r.get("best_target"),
            "Selectivity": r.get("selectivity_score"),
            "Hits": ",".join(r.get("target_hits", [])),
        })
    return pd.DataFrame(rows)


def load_examples(base: str = "outputs") -> Tuple[pd.DataFrame, pd.DataFrame]:
    db_df = _examples_to_df(f"{base}/predictor_db_examples.json")
    novel_df = _examples_to_df(f"{base}/predictor_novel_examples.json")
    return db_df, novel_df


def outputs_exist(base: str = "outputs") -> Dict[str, bool]:
    files = [
        "evaluation_report.json",
        "fingerprint_comparison.csv",
        "scaffold_ood_report.json",
        "run_summary.json",
        "predictor_db_examples.json",
        "predictor_novel_examples.json",
        "calibration_plot.png",
    ]
    return {f: Path(f"{base}/{f}").exists() for f in files}
