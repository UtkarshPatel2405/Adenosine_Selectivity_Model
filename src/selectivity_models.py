import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

from src.data_loader import load_and_clean
from src.features import build_feature_matrix, build_features
from src.scaffold_split import split_smiles_globally
from src.predictor import _load_scaler

SUBTYPES = ["A1", "A2A", "A2B", "A3"]


def build_selectivity_models(
    data_path: str = "data/raw/AR_all_unique_parents_with_smiles.csv",
    min_paired: int = 50,
):
    print("\n" + "=" * 60)
    print("TRAINING DIRECT SELECTIVITY MODELS (Delta-pChEMBL)")
    print("=" * 60)

    # Load the production scaler pipeline fitted globally on all training data
    try:
        pipeline = _load_scaler("precise")
        print("[INFO] Successfully loaded the global production scaler pipeline.")
    except Exception as e:
        print(
            f"[WARNING] Global scaler not found: {e}. Falling back to fitting scaler locally on each pair."
        )
        pipeline = None

    # 1. Load the processed DB lookup
    # Make sure we have the latest database lookup json
    _, lookup = load_and_clean(data_path, mode="precise", include_decoys=True)

    # 2. Identify paired compounds for each pair
    pairs = [
        ("A2A", "A1"),  # Focus on clinically important pairs
        ("A2A", "A3"),
        ("A1", "A3"),
        ("A1", "A2B"),
        ("A2A", "A2B"),
        ("A2B", "A3"),
    ]

    out_dir = Path("models/selectivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    selectivity_summary = {}

    for subA, subB in pairs:
        pair_name = f"{subA}_vs_{subB}"
        print(f"\nAnalyzing selectivity pair: {subA} vs {subB}...")

        # Collect smiles and values
        paired_data = []
        for smiles, values in lookup.items():
            if subA in values and subB in values:
                valA = values[subA]
                valB = values[subB]
                paired_data.append(
                    {
                        "smiles": smiles,
                        f"pchembl_{subA}": valA,
                        f"pchembl_{subB}": valB,
                        "delta_pchembl": valA - valB,
                    }
                )

        n_paired = len(paired_data)
        print(f"  Found {n_paired} paired compounds with values for both targets.")

        if n_paired < min_paired:
            print(
                f"  [SKIP] Too few paired compounds (< {min_paired}). Skipping direct model."
            )
            continue

        df_pair = pd.DataFrame(paired_data)

        # 3. Scaffold split (80-20) globally at the molecule level
        train_smiles, test_smiles = split_smiles_globally(
            df_pair["smiles"].unique(), test_size=0.2, random_state=42
        )
        train_df = df_pair[df_pair["smiles"].isin(train_smiles)].reset_index(drop=True)
        test_df = df_pair[df_pair["smiles"].isin(test_smiles)].reset_index(drop=True)

        # 4. Build features using the unified global pipeline
        if pipeline is not None:
            print("  Featurizing compounds using the global production pipeline...")
            X_train = np.vstack(
                [build_features(s, pipeline) for s in train_df["smiles"]]
            )
            X_test = np.vstack([build_features(s, pipeline) for s in test_df["smiles"]])
            pair_pipeline = pipeline
        else:
            print(
                "  [WARNING] Falling back to fitting pair-specific scaler pipeline..."
            )
            X_train, X_test, pair_pipeline = build_feature_matrix(
                train_df, test_df, smiles_col="smiles", save_to_disk=False
            )

        y_train = train_df["delta_pchembl"].values
        y_test = test_df["delta_pchembl"].values

        # 5. Train XGBoost regressor
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # 6. Evaluate
        preds = model.predict(X_test)
        r2 = float(r2_score(y_test, preds))
        mae = float(np.mean(np.abs(y_test - preds)))
        rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))

        print(f"  Evaluation Results for {pair_name}:")
        print(f"    R² Score: {r2:.4f}")
        print(f"    MAE:      {mae:.4f} pChEMBL units")
        print(f"    RMSE:     {rmse:.4f} pChEMBL units")

        # Save model and pipeline
        model_path = out_dir / f"xgb_selectivity_{pair_name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        pipeline_path = out_dir / f"xgb_selectivity_{pair_name}_pipeline.pkl"
        with open(pipeline_path, "wb") as f:
            pickle.dump(pair_pipeline, f)

        selectivity_summary[pair_name] = {
            "n_paired": n_paired,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        }

    summary_file = out_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(selectivity_summary, f, indent=2)
    print(f"\n[SUCCESS] Wrote direct selectivity summary to {summary_file}")


def predict_direct_selectivity(smiles: str, subA: str, subB: str) -> float | None:
    """
    Predict direct selectivity difference (pChEMBL_subA - pChEMBL_subB) for a given compound SMILES.
    """
    pair_name = f"{subA}_vs_{subB}"
    model_path = Path(f"models/selectivity/xgb_selectivity_{pair_name}_model.pkl")
    pipeline_path = Path(f"models/selectivity/xgb_selectivity_{pair_name}_pipeline.pkl")

    if not model_path.exists() or not pipeline_path.exists():
        # Fallback to reverse pair name if exists
        reverse_pair_name = f"{subB}_vs_{subA}"
        rev_model_path = Path(
            f"models/selectivity/xgb_selectivity_{reverse_pair_name}_model.pkl"
        )
        rev_pipeline_path = Path(
            f"models/selectivity/xgb_selectivity_{reverse_pair_name}_pipeline.pkl"
        )

        if rev_model_path.exists() and rev_pipeline_path.exists():
            with open(rev_model_path, "rb") as f:
                model = pickle.load(f)
            with open(rev_pipeline_path, "rb") as f:
                pipeline = pickle.load(f)
            x = build_features(smiles, pipeline)
            pred = model.predict(x.reshape(1, -1))[0]
            return -float(pred)  # Negate since order is reversed
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    x = build_features(smiles, pipeline)
    pred = model.predict(x.reshape(1, -1))[0]
    return float(pred)


if __name__ == "__main__":
    build_selectivity_models()
