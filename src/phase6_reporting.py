from __future__ import annotations

import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_and_clean

try:
    from src.data_splitter import scaffold_split
except Exception:
    from src.scaffold_split import scaffold_split


from src.predictor import SUBTYPES

def _ensure_dir(p: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: str, obj) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _murcko_scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False) or ""
    except Exception:
        return ""


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
        out.append(
            {
                "bin": i,
                "n": int(len(idx)),
                "std_mean": float(np.mean(y_std[idx])),
                "mae_mean": float(np.mean(np.abs(y_true[idx] - y_pred[idx]))),
            }
        )
    return out


def _load_local_models():
    """Loads the newly trained XGBoost models."""
    models = {}
    for st in SUBTYPES:
        filename = f"models/xgboost_{st.lower()}_model.pkl"
        try:
            with open(filename, "rb") as f:
                models[st] = pickle.load(f)
        except FileNotFoundError:
            print(f"WARNING: Could not find {filename}")
    return models

def _safe_predict(model, x: np.ndarray) -> Tuple[float, float]:
    """Handles both single models and ensembles safely."""
    if isinstance(model, (list, tuple)):
        preds = np.array([float(m.predict(x.reshape(1, -1))[0]) for m in model])
        return float(preds.mean()), float(preds.std(ddof=0))
    else:
        pred = float(model.predict(x.reshape(1, -1))[0])
        return pred, 0.0

_MORGAN = GetMorganGenerator(radius=2, fpSize=2048)  # ECFP4-like
_RDKFP = GetRDKitFPGenerator(fpSize=2048)

_DESC_NAMES = ["MW", "LogP", "HBD", "HBA", "RotBonds", "AromRings", "TPSA"]


def _descriptors_from_mol(mol: Chem.Mol) -> np.ndarray:
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    arom = Lipinski.NumAromaticRings(mol)
    tpsa = Descriptors.TPSA(mol)
    return np.array([mw, logp, hbd, hba, rot, arom, tpsa], dtype=float)


def _fp_to_np(fp, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _build_X(
    df,
    smiles_col: str,
    kind: str,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> Tuple[np.ndarray, StandardScaler | None]:
    fps = []
    descs = []

    for smi in df[smiles_col].tolist():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES in dataframe: {smi}")

        if kind == "morgan_only" or kind == "morgan_desc":
            fp = _MORGAN.GetFingerprint(mol)
        elif kind == "rdkit_only":
            fp = _RDKFP.GetFingerprint(mol)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        fps.append(_fp_to_np(fp, 2048))

        if kind == "morgan_desc":
            descs.append(_descriptors_from_mol(mol))

    X_fp = np.vstack(fps).astype(np.uint8)

    if kind != "morgan_desc":
        return X_fp.astype(float), scaler

    X_desc = np.vstack(descs).astype(float)

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        scaler.fit(X_desc)

    X_desc_scaled = scaler.transform(X_desc)
    X = np.hstack([X_fp.astype(float), X_desc_scaled])
    return X, scaler

def _evaluate_per_subtype_ensemble(train_df, test_df, X_train, X_test, y_train, y_test) -> dict:
    models = _load_local_models()

    per_subtype = {}
    all_true = []
    all_pred = []
    all_std = []

    for st in SUBTYPES:
        train_mask = (train_df["target_subtype"].values == st)
        test_mask = (test_df["target_subtype"].values == st)

        Xtr = X_train[train_mask]
        ytr = y_train[train_mask]
        Xte = X_test[test_mask]
        yte = y_test[test_mask]

        if len(yte) == 0 or st not in models:
            per_subtype[st] = {"n_test": 0, "skipped": True}
            continue

        preds = np.zeros((len(yte),), dtype=float)
        stds = np.zeros((len(yte),), dtype=float)
        for i in range(len(yte)):
            m, s = _safe_predict(models[st], Xte[i])
            preds[i] = m
            stds[i] = s

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(Xtr, ytr)
        base_preds = baseline.predict(Xte)

        per_subtype[st] = {
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "model_mae": float(mean_absolute_error(yte, preds)),
            "model_rmse": _rmse(yte, preds),
            "model_r2": float(r2_score(yte, preds)),
            "baseline_mae": float(mean_absolute_error(yte, base_preds)),
            "baseline_rmse": _rmse(yte, base_preds),
            "baseline_r2": float(r2_score(yte, base_preds)),
            "delta_mae": float(mean_absolute_error(yte, preds) - mean_absolute_error(yte, base_preds)),
            "uncertainty_mean_std": float(np.mean(stds)),
            "calibration_quartiles": _calibration_quartiles(yte, preds, stds),
        }

        all_true.append(yte)
        all_pred.append(preds)
        all_std.append(stds)

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    y_std = np.concatenate(all_std) if all_std else np.array([])

    base_all = DummyRegressor(strategy="mean")
    base_all.fit(X_train, y_train)
    base_pred_all = base_all.predict(X_test)

    overall = {
        "model_mae": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else None,
        "model_rmse": _rmse(y_true, y_pred) if len(y_true) else None,
        "model_r2": float(r2_score(y_true, y_pred)) if len(y_true) else None,
        "baseline_mae": float(mean_absolute_error(y_test, base_pred_all)),
        "baseline_rmse": _rmse(y_test, base_pred_all),
        "baseline_r2": float(r2_score(y_test, base_pred_all)),
        "calibration_quartiles": _calibration_quartiles(y_true, y_pred, y_std) if len(y_true) else [],
    }

    return {"overall": overall, "per_subtype": per_subtype}


def _plot_calibration(calibration_bins: List[dict], out_path: str) -> None:
    _ensure_dir(out_path)

    if not calibration_bins:
        plt.figure(figsize=(6, 4))
        plt.title("Uncertainty calibration (quartiles)")
        plt.text(0.5, 0.5, "Not enough data for calibration bins", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return

    xs = [b["std_mean"] for b in calibration_bins]
    ys = [b["mae_mean"] for b in calibration_bins]
    ns = [b["n"] for b in calibration_bins]

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    for x, y, n in zip(xs, ys, ns):
        plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

    plt.xlabel("Predicted uncertainty (ensemble std) — mean per quartile")
    plt.ylabel("Absolute error — mean per quartile")
    plt.title("Uncertainty calibration (higher std should mean higher error)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _scaffold_ood_report(train_df, test_df, y_test: np.ndarray, y_pred_test: np.ndarray) -> dict:
    train_scaffolds = set(train_df["canonical_smiles"].apply(_murcko_scaffold).tolist())
    test_scaff = test_df["canonical_smiles"].apply(_murcko_scaffold).tolist()
    is_seen = np.array([s in train_scaffolds and s != "" for s in test_scaff], dtype=bool)

    def _metrics(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {"n": 0, "mae": None, "rmse": None, "r2": None}
        yt = y_test[mask]
        yp = y_pred_test[mask]
        return {
            "n": int(mask.sum()),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": _rmse(yt, yp),
            "r2": float(r2_score(yt, yp)),
        }

    return {
        "seen_scaffold": _metrics(is_seen),
        "novel_scaffold": _metrics(~is_seen),
    }


def _fingerprint_comparison(train_df, test_df, random_state: int, out_csv: str) -> List[dict]:
    import xgboost as xgb

    y_train = train_df["pchembl_value"].to_numpy(float)
    y_test = test_df["pchembl_value"].to_numpy(float)
    rows = []

    Xtr, _ = _build_X(train_df, "canonical_smiles", kind="morgan_only")
    Xte, _ = _build_X(test_df, "canonical_smiles", kind="morgan_only")

    Xtr_rdk, _ = _build_X(train_df, "canonical_smiles", kind="rdkit_only")
    Xte_rdk, _ = _build_X(test_df, "canonical_smiles", kind="rdkit_only")

    scaler = StandardScaler()
    Xtr_md, scaler = _build_X(train_df, "canonical_smiles", kind="morgan_desc", scaler=scaler, fit_scaler=True)
    Xte_md, _ = _build_X(test_df, "canonical_smiles", kind="morgan_desc", scaler=scaler, fit_scaler=False)

    feature_sets = [
        ("Morgan2048_only", Xtr, Xte),
        ("RDKitFP2048_only", Xtr_rdk, Xte_rdk),
        ("Morgan2048_plus_7desc", Xtr_md, Xte_md),
    ]

    xgb_params = dict(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    for name, Xtr_i, Xte_i in feature_sets:
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(Xtr_i, y_train)
        pred = model.predict(Xte_i)
        rows.append(
            {
                "feature_set": name,
                "model": "XGBRegressor",
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": _rmse(y_test, pred),
                "r2": float(r2_score(y_test, pred)),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
            }
        )

    _ensure_dir(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return rows

def run_phase6_mode(mode: str, data_path: str, test_size: float, random_state: int):
    print(f"\n--- Running Phase 6 Reporting for Mode: {mode.upper()} ---")
    
    # 1) Load + split
    df, _lookup = load_and_clean(data_path, mode=mode)
    train_df, test_df = scaffold_split(df, test_size=test_size, random_state=random_state, smiles_col="canonical_smiles")

    y_train = train_df["pchembl_value"].to_numpy(float)
    y_test = test_df["pchembl_value"].to_numpy(float)

    # 2) Evaluate models
    scaler = StandardScaler()
    X_train, scaler = _build_X(train_df, "canonical_smiles", kind="morgan_desc", scaler=scaler, fit_scaler=True)
    X_test, _ = _build_X(test_df, "canonical_smiles", kind="morgan_desc", scaler=scaler, fit_scaler=False)

    eval_report = _evaluate_per_subtype_ensemble(train_df, test_df, X_train, X_test, y_train, y_test)
    evaluation_payload = {
        "mode": mode,
        "data_path": data_path,
        "split": {"test_size": test_size, "random_state": random_state},
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        **eval_report,
    }
    
    out_dir = f"outputs/validoutput/{mode}"
    _ensure_dir(out_dir)
    
    _write_json(f"{out_dir}/evaluation_report.json", evaluation_payload)

    
    _plot_calibration(evaluation_payload["overall"]["calibration_quartiles"], f"{out_dir}/calibration_plot.png")

   
    models = _load_local_models()
    y_pred_all = np.zeros((len(test_df),), dtype=float)
    y_std_all = np.zeros((len(test_df),), dtype=float)

    for i in range(len(test_df)):
        st = test_df.iloc[i]["target_subtype"]
        if st in models:
            m, s = _safe_predict(models[st], X_test[i])
            y_pred_all[i] = m
            y_std_all[i] = s

    ood = _scaffold_ood_report(train_df, test_df, y_test=y_test, y_pred_test=y_pred_all)
    ood_payload = {
        "mode": mode,
        "data_path": data_path,
        "split": {"test_size": test_size, "random_state": random_state},
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "ood_by_scaffold": ood,
    }
    _write_json(f"{out_dir}/scaffold_ood_report.json", ood_payload)

   
    fp_rows = _fingerprint_comparison(
        train_df=train_df,
        test_df=test_df,
        random_state=random_state,
        out_csv=f"{out_dir}/fingerprint_comparison.csv",
    )

    if evaluation_payload["overall"]["model_mae"] is not None:
        print(
            f"Overall MAE: {evaluation_payload['overall']['model_mae']:.4f} | "
            f"Baseline MAE: {evaluation_payload['overall']['baseline_mae']:.4f}"
        )


def main():
    data_path = "data/raw/AR_all_unique_parents_with_smiles.csv"
    test_size = 0.2
    random_state = 42

    for mode in ["standard", "strict"]:
        run_phase6_mode(mode, data_path, test_size, random_state)

if __name__ == "__main__":
    main()