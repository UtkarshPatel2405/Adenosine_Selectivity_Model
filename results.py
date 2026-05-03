import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data_loader import load_and_clean

try:
    from src.data_splitter import scaffold_split
except Exception:
    from src.scaffold_split import scaffold_split

from src.features import build_feature_matrix
from src.predictor import predict


def _write_json(path: str, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def run_mode(
    mode: str,
    out_dir: str,
    data_path: str = "data/raw/AR_all_unique_parents_with_smiles.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    hit_threshold: float = 6.0,
    n_db_examples: int = 5,
):
    started = datetime.now().isoformat(timespec="seconds")

    # 1) Data
    df, lookup = load_and_clean(data_path, mode=mode)

    # 2) Split
    train_df, test_df = scaffold_split(df, test_size=test_size, random_state=random_state)

    # 3) Features (also writes caches/scaler per your features.py)
    X_train, X_test, _scaler = build_feature_matrix(train_df, test_df, smiles_col="canonical_smiles")

    # 4) Predictor examples
    db_smiles = df["canonical_smiles"].dropna().unique().tolist()[:n_db_examples]
    db_results = [{"smiles": s, "result": predict(s, threshold=hit_threshold)} for s in db_smiles]

    novel_smiles = [
        "c1ccc2[nH]ccc2c1",
        "CCN(CC)CCc1ccc2[nH]ccc2c1",
        "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
        "CC(=O)Nc1ccc(O)cc1",
    ]
    novel_results = []
    for s in novel_smiles:
        try:
            r = predict(s, threshold=hit_threshold)
            novel_results.append({"smiles": s, "result": r})
        except Exception as e:
            novel_results.append({"smiles": s, "error": str(e)})

    # 5) Summary
    summary = {
        "started_at": started,
        "mode": mode,
        "data_path": data_path,
        "n_rows_clean": int(len(df)),
        "n_lookup_smiles": int(len(lookup)),
        "split": {
            "test_size": test_size,
            "random_state": random_state,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        },
        "features": {
            "X_train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
            "X_test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
            "fingerprint_bits": 2048,
            "n_descriptors": 7,
        },
        "sanity": {
            "fp_unique_values": sorted(list(set(np.unique(X_train[:, :2048]).tolist()))),
            "desc_mean": float(X_train[:, 2048:].mean()),
        },
        "artifacts_expected": {
            "db_lookup": "data/processed/db_lookup.json",
            "scaler": "models/scaler.pkl",
            "features_train": "data/processed/features_train.pkl",
            "features_test": "data/processed/features_test.pkl",
            "train_fps": "data/processed/train_fps.pkl",
            "models": [
                "models/xgb_A1_ens.pkl",
                "models/xgb_A2A_ens.pkl",
                "models/xgb_A2B_ens.pkl",
                "models/xgb_A3_ens.pkl",
                "models/xgb_global_ens.pkl",
            ],
        },
    }

    _write_json(f"{out_dir}/run_summary.json", summary)
    _write_json(f"{out_dir}/predictor_db_examples.json", db_results)
    _write_json(f"{out_dir}/predictor_novel_examples.json", novel_results)

    print(f"PASS: wrote outputs to {out_dir}/")


def main():
    base_out = "outputs/validoutput"
    for mode in ["standard", "strict"]:
        run_mode(mode=mode, out_dir=f"{base_out}/{mode}")


if __name__ == "__main__":
    main()
