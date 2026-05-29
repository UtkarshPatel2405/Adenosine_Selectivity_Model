# src/train_antagonists.py
import os
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

from src.data_loader import load_and_clean
from src.features import build_feature_matrix
from src.conformal import train_conformal_model
from src.scaffold_split import scaffold_split
from src.pcm_model import train_pcm_production_model

SUBTYPES = ["A1", "A2A", "A2B", "A3"]

# Scientific hyperparameter configs
PARAMS_KI = {
    "A1": {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A2A": {"n_estimators": 700, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A2B": {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A3": {"n_estimators": 700, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2}
}

PARAMS_IC50 = {
    "A1": {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A2A": {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A2B": {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A3": {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2}
}

def train_mode_pipeline(
    role: str,
    endpoint: str,
    output_folder: str,
    data_path: str = "data/raw/AR_all_unique_parents_with_smiles.csv"
):
    print("\n" + "="*70)
    print(f"TRAINING SUITE: Role={role.upper()}, Endpoint={endpoint.upper()} -> Folder={output_folder}")
    print("="*70)
    
    # 1. Load clean filtered data with mutual decoy ingestion
    lookup_name = f"db_lookup_{role}_{endpoint.lower()}.json"
    save_lookup_path = f"data/processed/{lookup_name}"
    
    df, _ = load_and_clean(
        data_path,
        save_lookup_path=save_lookup_path,
        mode="precise",
        target_role=role,
        target_endpoint=endpoint,
        include_decoys=True
    )
    df = df.rename(columns={"canonical_smiles": "smiles"})
    
    if len(df) < 100:
        print(f"[SKIP] Too few compounds available for {role} ({endpoint}): {len(df)}")
        return
        
    # 2. Scaffold Split to check generalizability (memorizing vs learning test)
    train_df, test_df = scaffold_split(df, test_size=0.2, random_state=42, smiles_col="smiles")
    
    # 3. Build features globally on this split
    X_train_glob, X_test_glob, pipeline = build_feature_matrix(train_df, test_df, smiles_col="smiles", save_to_disk=False)
    
    # Save scalers
    os.makedirs(output_folder, exist_ok=True)
    scaler_path = Path(output_folder) / f"scaler_{role}_{endpoint.lower()}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[SUCCESS] Saved scaler to {scaler_path}")
    
    # 4. Train each subtype model
    configs = PARAMS_KI if endpoint.lower() in ["ki", "kd"] else PARAMS_IC50
    metrics_summary = {}
    
    for st in SUBTYPES:
        print(f"\nTraining Model for Subtype {st}...")
        
        train_mask = (train_df["target_subtype"] == st).values
        test_mask = (test_df["target_subtype"] == st).values
        
        X_tr = X_train_glob[train_mask]
        y_tr = train_df.loc[train_mask, "pchembl_value"].values
        
        X_te = X_test_glob[test_mask]
        y_te = test_df.loc[test_mask, "pchembl_value"].values
        
        if len(y_tr) < 30:
            print(f"  [SKIP] Too few training samples for {st} ({len(y_tr)})")
            continue
            
        print(f"  Samples: Train={len(y_tr)} (including decoys), Test={len(y_te)}")
        
        # Load params for this subtype
        params = configs.get(st, {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05}).copy()
        params.update({"tree_method": "hist", "n_jobs": -1, "random_state": 42})
        
        # Train MAPIE conformal regressor
        base_xgb = xgb.XGBRegressor(**params)
        conformal_model = train_conformal_model(base_xgb, X_tr, y_tr, cv=5)
        
        # Save model
        model_name = f"xgboost_{role}_{endpoint.lower()}_{st.lower()}_model.pkl"
        model_path = Path(output_folder) / model_name
        with open(model_path, "wb") as f:
            pickle.dump(conformal_model, f)
        print(f"  [SUCCESS] Saved model to {model_path}")
        
        # Evaluate performance on scaffold test set (OOD validation check)
        from sklearn.metrics import r2_score
        preds = conformal_model.predict(X_te)
        r2 = float(r2_score(y_te, preds))
        mae = float(np.mean(np.abs(y_te - preds)))
        rmse = float(np.sqrt(np.mean((y_te - preds) ** 2)))
        
        print(f"  [METRICS] {st} Scaffold OOD Validation:")
        print(f"    R² Score: {r2:.4f}")
        print(f"    MAE:      {mae:.4f}")
        print(f"    RMSE:     {rmse:.4f}")
        
        metrics_summary[st] = {"r2": r2, "mae": mae, "rmse": rmse}
        
    summary_path = Path(output_folder) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n[SUCCESS] Completed Suite. Evaluation saved to {summary_path}")
    
    return metrics_summary, df, train_df, test_df

def run_antagonist_suite():
    data_path = "data/raw/AR_all_unique_parents_with_smiles.csv"
    
    # 1. Antagonist pKi models
    train_mode_pipeline(
        role="antagonist",
        endpoint="Ki",
        output_folder="models/antagonist_ki",
        data_path=data_path
    )
    
    # 2. Antagonist pIC50 models
    _, pcm_df, pcm_train_df, pcm_test_df = train_mode_pipeline(
        role="antagonist",
        endpoint="IC50",
        output_folder="models/antagonist_ic50",
        data_path=data_path
    )
    
    # 3. Train unified PCM model (on full antagonist database)
    print("\n" + "="*70)
    print("TRAINING SUITE: UNIFIED PROTEOCHEMOMETRIC (PCM) MODEL")
    print("="*70)
    train_pcm_production_model(pcm_train_df, pcm_test_df)

if __name__ == "__main__":
    run_antagonist_suite()
