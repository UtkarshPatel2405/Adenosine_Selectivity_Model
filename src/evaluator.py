"""
Evaluator — Generate test set statistics with REAL conformal prediction intervals.

Fixes the broken std_mean=0.0 issue by properly calling predict_interval() on conformal models.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_and_clean
from src.predictor import (
    _load_xgb_models,
    _load_rf_models,
    SUBTYPES,
)


def _write_json(path: str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _calibration_quartiles(
    y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray
) -> List[dict]:
    """
    Group predictions into quartiles by predicted uncertainty.
    A well-calibrated model shows monotonically increasing MAE as uncertainty increases.
    """
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


def evaluate(
    mode: str = "precise",
    data_path: str = "data/raw",
    test_size: float = 0.2,
    random_state: int = 42,
    out_path: str | None = None,
    include_decoys: bool = True,
) -> dict:
    """
    Evaluate models with real conformal prediction intervals.

    Key fix: Ensures _ensemble_predict extracts actual uncertainty from
    CrossConformalRegressor.predict_interval() instead of returning 0.0.
    """
    if out_path is None:
        out_path = f"outputs/validoutput/{mode}/evaluation_{mode}_report.json"

    # Load data with same settings as training
    df, _ = load_and_clean(data_path, mode=mode, include_decoys=include_decoys)

    # Load global scaffold split from production training
    split_path = Path("data/processed/global_split.json")
    with open(split_path) as f:
        split = json.load(f)
    train_smiles = set(split["train"])
    test_smiles = set(split["test"])

    # Filter to SMILES present in our data (handles any data drift)
    available_smiles = set(df["canonical_smiles"])
    train_smiles = train_smiles & available_smiles
    test_smiles = test_smiles & available_smiles

    train_df = df[df["canonical_smiles"].isin(train_smiles)].reset_index(drop=True)
    test_df = df[df["canonical_smiles"].isin(test_smiles)].reset_index(drop=True)

    # Load production scaler pipeline
    import pickle
    from src.features import _morgan_bits, _maccs_bits, _all_descriptors

    scaler_path = Path(f"models/{mode}/scaler_{mode}.pkl")
    if not scaler_path.exists():
        scaler_path = Path("models/scaler.pkl")

    print(f"  [INFO] Loading production feature pipeline from {scaler_path}")
    with open(scaler_path, "rb") as f:
        pipeline = pickle.load(f)

    def transform_df(df):
        from joblib import Parallel, delayed

        smiles = df["canonical_smiles"].tolist()
        Xfp = np.vstack(Parallel(n_jobs=-1)(delayed(_morgan_bits)(s) for s in smiles))
        Xmaccs = np.vstack(Parallel(n_jobs=-1)(delayed(_maccs_bits)(s) for s in smiles))
        Xdesc = np.vstack(
            Parallel(n_jobs=-1)(delayed(_all_descriptors)(s) for s in smiles)
        )
        Xdesc_s = pipeline.transform(Xdesc)
        return np.hstack([Xfp, Xmaccs, Xdesc_s]).astype(np.float32)

    X_train = transform_df(train_df)
    X_test = transform_df(test_df)

    y_train_all = train_df["pchembl_value"].to_numpy(dtype=float)
    y_test_all = test_df["pchembl_value"].to_numpy(dtype=float)

    models = _load_xgb_models()
    rf_models = _load_rf_models()

    per_subtype: Dict[str, dict] = {}
    all_preds = []
    all_true = []
    all_std = []
    all_lowers = []
    all_uppers = []

    for st in SUBTYPES:
        train_mask = train_df["target_subtype"].values == st
        test_mask = test_df["target_subtype"].values == st

        Xtr = X_train[train_mask]
        ytr = y_train_all[train_mask]
        Xte = X_test[test_mask]
        yte = y_test_all[test_mask]

        if len(yte) == 0:
            per_subtype[st] = {"n_test": 0, "skipped": True}
            continue

        model_ens = models[st]
        if type(model_ens).__name__ in ("CrossConformalRegressor", "MapieRegressor"):
            if type(model_ens).__name__ == "CrossConformalRegressor":
                y_pred, y_pis = model_ens.predict_interval(Xte)
                if y_pis.ndim == 3:
                    lowers = y_pis[:, 0, 0]
                    uppers = y_pis[:, 1, 0]
                else:
                    lowers = y_pis[:, 0]
                    uppers = y_pis[:, 1]
            else:  # MapieRegressor
                y_pred, y_pis = model_ens.predict(Xte, alpha=0.10)
                if y_pis.ndim == 3:
                    lowers = y_pis[:, 0, 0]
                    uppers = y_pis[:, 1, 0]
                else:
                    lowers = y_pis[:, 0]
                    uppers = y_pis[:, 1]
            stds = (uppers - lowers) / 3.29
            preds = y_pred
        elif isinstance(model_ens, (list, tuple)):
            sub_preds = np.array([m.predict(Xte) for m in model_ens])
            preds = np.mean(sub_preds, axis=0)
            stds = np.std(sub_preds, axis=0)
            lowers = preds - 1.96 * stds
            uppers = preds + 1.96 * stds
        else:
            preds = model_ens.predict(Xte)
            stds = np.zeros_like(preds)
            lowers = preds
            uppers = preds

        # Verify conformal intervals are producing real uncertainty
        mean_std = float(np.mean(stds))
        if mean_std < 1e-6:
            print(
                f"  [WARNING] {st}: Conformal uncertainty is near-zero ({mean_std:.6f}). Model may not be a conformal wrapper."
            )
        else:
            print(f"  [OK] {st}: Mean conformal uncertainty = {mean_std:.4f}")

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(Xtr, ytr)
        base_preds = baseline.predict(Xte)

        # Check conformal coverage
        in_interval = np.sum((yte >= lowers) & (yte <= uppers))
        coverage = float(in_interval / len(yte)) if len(yte) > 0 else 0.0

        report_st = {
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "model_mae": float(mean_absolute_error(yte, preds)),
            "model_rmse": float(np.sqrt(mean_squared_error(yte, preds))),
            "model_r2": float(r2_score(yte, preds)),
            "baseline_mae": float(mean_absolute_error(yte, base_preds)),
            "baseline_rmse": float(np.sqrt(mean_squared_error(yte, base_preds))),
            "baseline_r2": float(r2_score(yte, base_preds)),
            "delta_mae": float(
                mean_absolute_error(yte, preds) - mean_absolute_error(yte, base_preds)
            ),
            "uncertainty_mean_std": mean_std,
            "conformal_coverage_90": coverage,
            "calibration_quartiles": _calibration_quartiles(yte, preds, stds),
        }

        # Evaluate GNN directly on unified Test Set
        try:
            from src.gnn_model import MoleculeGNN, smiles_to_graph
            import torch
            from torch_geometric.loader import DataLoader

            gnn_model_path = Path(f"models/gnn/gnn_{st.lower()}_model.pt")
            if gnn_model_path.exists():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(
                    gnn_model_path, map_location=device, weights_only=False
                )
                gnn_model = MoleculeGNN(
                    node_dim=checkpoint.get("node_dim", 140),
                    edge_dim=checkpoint.get("edge_dim", 7),
                    hidden_dim=checkpoint.get("hidden_dim", 256),
                    num_layers=checkpoint.get("num_layers", 3),
                ).to(device)
                gnn_model.load_state_dict(checkpoint["model_state_dict"])
                gnn_model.eval()

                test_smiles_st = test_df.loc[test_mask, "canonical_smiles"].tolist()
                test_graphs = []
                for idx, smi in enumerate(test_smiles_st):
                    g = smiles_to_graph(smi)
                    if g is not None:
                        g.y = torch.tensor([yte[idx]], dtype=torch.float)
                        test_graphs.append(g)

                if test_graphs:
                    loader = DataLoader(test_graphs, batch_size=128, shuffle=False)
                    preds_gnn = []
                    true_gnn = []
                    with torch.no_grad():
                        for data in loader:
                            data = data.to(device)
                            out = gnn_model(data)
                            preds_gnn.extend(out.cpu().numpy().tolist())
                            true_gnn.extend(data.y.cpu().numpy().tolist())

                    report_st["gnn_mae"] = float(
                        mean_absolute_error(true_gnn, preds_gnn)
                    )
                    report_st["gnn_rmse"] = float(
                        np.sqrt(mean_squared_error(true_gnn, preds_gnn))
                    )
                    report_st["gnn_r2"] = float(r2_score(true_gnn, preds_gnn))
                else:
                    report_st["gnn_mae"] = None
                    report_st["gnn_rmse"] = None
                    report_st["gnn_r2"] = None
            else:
                report_st["gnn_mae"] = None
                report_st["gnn_rmse"] = None
                report_st["gnn_r2"] = None
        except Exception as e:
            print(f"  [WARNING] GNN evaluation failed for {st}: {e}")
            report_st["gnn_mae"] = None
            report_st["gnn_rmse"] = None
            report_st["gnn_r2"] = None

        # Add Random Forest metrics
        rf_ens = rf_models.get(st)
        if rf_ens:
            rf_preds = rf_ens.predict(Xte)
            report_st["rf_mae"] = float(mean_absolute_error(yte, rf_preds))
            report_st["rf_rmse"] = float(np.sqrt(mean_squared_error(yte, rf_preds)))
            report_st["rf_r2"] = float(r2_score(yte, rf_preds))
        else:
            report_st["rf_mae"] = None
            report_st["rf_rmse"] = None
            report_st["rf_r2"] = None

        per_subtype[st] = report_st

        all_preds.append(preds)
        all_true.append(yte)
        all_std.append(stds)
        all_lowers.append(lowers)
        all_uppers.append(uppers)

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_std = np.concatenate(all_std) if all_std else np.array([])

    base_all = DummyRegressor(strategy="mean")
    base_all.fit(X_train, y_train_all)
    base_pred_all = base_all.predict(X_test)

    # Overall conformal coverage
    overall_coverage = 0.0
    if len(y_true) > 0 and len(all_lowers) > 0:
        y_lower = np.concatenate(all_lowers)
        y_upper = np.concatenate(all_uppers)
        overall_coverage = float(
            np.sum((y_true >= y_lower) & (y_true <= y_upper)) / len(y_true)
        )

    summary = {
        "mode": mode,
        "data_path": data_path,
        "include_decoys": include_decoys,
        "split": {"test_size": test_size, "random_state": random_state},
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "overall": {
            "model_mae": float(mean_absolute_error(y_true, y_pred))
            if len(y_true)
            else None,
            "model_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))
            if len(y_true)
            else None,
            "model_r2": float(r2_score(y_true, y_pred)) if len(y_true) else None,
            "baseline_mae": float(mean_absolute_error(y_test_all, base_pred_all)),
            "baseline_rmse": float(
                np.sqrt(mean_squared_error(y_test_all, base_pred_all))
            ),
            "baseline_r2": float(r2_score(y_test_all, base_pred_all)),
            "conformal_coverage_90": overall_coverage,
            "calibration_quartiles": _calibration_quartiles(y_true, y_pred, y_std)
            if len(y_true)
            else [],
        },
        "per_subtype": per_subtype,
    }

    _write_json(out_path, summary)
    print(f"\n[SUCCESS] Evaluation report written to {out_path}")
    return summary


def evaluate_actives_only(mode: str = "precise", data_path: str = "data/raw") -> dict:
    """
    Separate evaluation on actives-only (no decoys) for honest reporting.
    This prevents inflated R² from easy-to-predict decoy compounds.
    """
    out_path = f"outputs/validoutput/{mode}/evaluation_{mode}_actives_only_report.json"
    return evaluate(
        mode=mode,
        data_path=data_path,
        out_path=out_path,
        include_decoys=False,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("EVALUATING WITH DECOYS (full dataset)")
    print("=" * 60)
    rep_full = evaluate(mode="precise")

    print("\n" + "=" * 60)
    print("EVALUATING ACTIVES-ONLY (honest baseline)")
    print("=" * 60)
    rep_actives = evaluate_actives_only(mode="precise")

    # Print comparison
    if (
        rep_full["overall"]["model_r2"] is not None
        and rep_actives["overall"]["model_r2"] is not None
    ):
        print(f"\n[COMPARISON] Full dataset R² = {rep_full['overall']['model_r2']:.4f}")
        print(
            f"[COMPARISON] Actives-only R² = {rep_actives['overall']['model_r2']:.4f}"
        )
        print("[INFO] Report BOTH in publications for transparency.")
