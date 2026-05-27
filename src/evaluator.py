import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_and_clean
from src.scaffold_split import scaffold_split

from src.features import build_feature_matrix
from src.predictor import _load_models, _ensemble_predict, SUBTYPES


def _write_json(path: str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _calibration_quartiles(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> List[dict]:
    n = len(y_true)
    if n < 8:
        return []

    order = np.argsort(y_std)
    bins = np.array_split(order, 4)

    out = []
    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        mae = float(np.mean(np.abs(y_true[idx] - y_pred[idx])))
        out.append(
            {
                "bin": i,
                "n": int(len(idx)),
                "std_mean": float(np.mean(y_std[idx])),
                "mae_mean": mae,
            }
        )
    return out


def evaluate(mode: str = "precise",
             data_path: str = "data/raw/AR_all_unique_parents_with_smiles.csv",
             test_size: float = 0.2,
             random_state: int = 42,
             out_path: str | None = None) -> dict:
    if out_path is None:
        infix = "precise" if mode == "precise" else "std" if mode == "standard" else "strict" if mode == "strict" else "root"
        out_path = f"outputs/validoutput/{mode}/evaluation_{infix}_report.json"
   
    df, _ = load_and_clean(data_path, mode=mode)
    train_df, test_df = scaffold_split(
    df, 
    test_size=test_size, 
    random_state=random_state, 
    smiles_col="canonical_smiles"  # Add this parameter
)
    X_train, X_test, _scaler = build_feature_matrix(train_df, test_df, smiles_col="canonical_smiles")

    y_train_all = train_df["pchembl_value"].to_numpy(dtype=float)
    y_test_all = test_df["pchembl_value"].to_numpy(dtype=float)

    models = _load_models(mode=mode)

    per_subtype: Dict[str, dict] = {}
    all_preds = []
    all_true = []
    all_std = []

    for st in SUBTYPES:
        train_mask = (train_df["target_subtype"].values == st)
        test_mask = (test_df["target_subtype"].values == st)

        Xtr = X_train[train_mask]
        ytr = y_train_all[train_mask]
        Xte = X_test[test_mask]
        yte = y_test_all[test_mask]

        if len(yte) == 0:
            per_subtype[st] = {"n_test": 0, "skipped": True}
            continue

        preds = np.zeros((len(yte),), dtype=float)
        stds = np.zeros((len(yte),), dtype=float)
        for i in range(len(yte)):
            m, s, _, _ = _ensemble_predict(models[st], Xte[i])
            preds[i] = m
            stds[i] = s

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(Xtr, ytr)
        base_preds = baseline.predict(Xte)

        report_st = {
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "model_mae": float(mean_absolute_error(yte, preds)),
            "model_rmse": float(np.sqrt(mean_squared_error(yte, preds))),
            "model_r2": float(r2_score(yte, preds)),
            "baseline_mae": float(mean_absolute_error(yte, base_preds)),
            "baseline_rmse": float(np.sqrt(mean_squared_error(yte, base_preds))),
            "baseline_r2": float(r2_score(yte, base_preds)),
            "delta_mae": float(mean_absolute_error(yte, preds) - mean_absolute_error(yte, base_preds)),
            "uncertainty_mean_std": float(np.mean(stds)),
            "calibration_quartiles": _calibration_quartiles(yte, preds, stds),
        }
        per_subtype[st] = report_st

        all_preds.append(preds)
        all_true.append(yte)
        all_std.append(stds)

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_std = np.concatenate(all_std) if all_std else np.array([])

    base_all = DummyRegressor(strategy="mean")
    base_all.fit(X_train, y_train_all)
    base_pred_all = base_all.predict(X_test)

    summary = {
        "mode": mode,
        "data_path": data_path,
        "split": {"test_size": test_size, "random_state": random_state},
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "overall": {
            "model_mae": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else None,
            "model_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else None,
            "model_r2": float(r2_score(y_true, y_pred)) if len(y_true) else None,
            "baseline_mae": float(mean_absolute_error(y_test_all, base_pred_all)),
            "baseline_rmse": float(np.sqrt(mean_squared_error(y_test_all, base_pred_all))),
            "baseline_r2": float(r2_score(y_test_all, base_pred_all)),
            "calibration_quartiles": _calibration_quartiles(y_true, y_pred, y_std) if len(y_true) else [],
        },
        "per_subtype": per_subtype,
    }

    _write_json(out_path, summary)
    return summary


if __name__ == "__main__":
    for mode in ["precise"]:
        rep = evaluate(mode=mode)
        infix = "precise" if mode == "precise" else "std" if mode == "standard" else "strict" if mode == "strict" else "root"
        print(f"Wrote outputs/{mode}/evaluation_{infix}_report.json")
        if rep["overall"]["model_mae"] is not None and rep["overall"]["model_mae"] < rep["overall"]["baseline_mae"]:
            print(f"PASS ({mode}): model beats baseline on overall MAE")
        else:
            print(f"FAIL ({mode}): model does not beat baseline on overall MAE")
