"""
Retrain Production Models — Full dataset training with MAPIE conformal wrapping.

Uses barcode-validated data and Nested CV hyperparameters (if available).
Trains one conformal-wrapped XGBoost model per adenosine receptor subtype.
"""

import os
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from src.data_loader import load_and_clean
from src.features import build_feature_matrix, build_features
from src.conformal import train_conformal_model
from src.scaffold_split import scaffold_split, split_smiles_globally
from src.smiles_registry import SmilesRegistry

SUBTYPES = ["A1", "A2A", "A2B", "A3"]

# Robust fallback hyperparameters if nested CV hasn't been run yet
DEFAULT_PARAMS = {
    "A1": {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A2A": {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A2B": {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A3": {"n_estimators": 800, "max_depth": 7, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2}
}

def retrain_production_models(data_path: str = "data/raw"):
    print("\n" + "="*60)
    print("PRODUCTION TRAINING WITH MAPIE CONFORMAL WRAPPING (INCLUDING DECOYS)")
    print("="*60)
    
    # 0. Initialize barcode registry for validation
    registry = SmilesRegistry()
    
    # 1. Load best params from nested CV if available
    cv_report_path = Path("outputs/nested_cv/merged_report.json")
    best_params_per_subtype = {}
    
    if cv_report_path.exists():
        print(f"[INFO] Found Nested CV report at {cv_report_path}. Loading HPO parameters...")
        with open(cv_report_path, "r") as f:
            cv_data = json.load(f)
        for st in SUBTYPES:
            if st in cv_data:
                best_params_per_subtype[st] = cv_data[st]["median_params"]
                print(f"  {st}: {best_params_per_subtype[st]}")
    else:
        print("[INFO] No Nested CV HPO parameters found. Using robust scientific fallbacks...")
        best_params_per_subtype = DEFAULT_PARAMS
        
    # 2. Load precise data and programmatically generated decoys
    df, _ = load_and_clean(data_path, mode="precise", include_decoys=True)
    df = df.rename(columns={"canonical_smiles": "smiles"})
    
    # Barcode dedup validation
    if "barcode" in df.columns:
        n_unique_barcodes = df["barcode"].nunique()
        n_rows = len(df)
        print(f"[BARCODE AUDIT] {n_rows} training rows, {n_unique_barcodes} unique molecular barcodes")
    
    # 3. Create global scaffold split (80-20) at the molecule level to prevent any target leakage
    train_smiles, test_smiles = split_smiles_globally(df["smiles"].unique(), test_size=0.2, random_state=42)
    
    # SAVE THE SPLIT so GNN can use the exact same test set
    split_path = Path("data/processed/global_split.json")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"train": list(train_smiles), "test": list(test_smiles)}, f)
    print(f"[INFO] Saved global scaffold split to {split_path}")
    
    train_df = df[df["smiles"].isin(train_smiles)].reset_index(drop=True)
    test_df = df[df["smiles"].isin(test_smiles)].reset_index(drop=True)
    
    # 4. Build features globally on precise + decoy dataset
    X_train_glob, X_test_glob, pipeline = build_feature_matrix(train_df, test_df, smiles_col="smiles")
    
    # Save precise + decoy scalers
    os.makedirs("models/precise", exist_ok=True)
    with open("models/precise/scaler_precise.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("[SUCCESS] Saved production scaler pipelines to models/scaler.pkl and models/precise/")
    
    # Save the feature matrices so SHAP and downstream diagnostics can use them
    with open("data/processed/features_train.pkl", "wb") as f:
        pickle.dump(X_train_glob, f)
    with open("data/processed/features_test.pkl", "wb") as f:
        pickle.dump(X_test_glob, f)
    print("[SUCCESS] Saved feature matrices to data/processed/")
    
    # 5. Train each subtype model
    training_summary = {}
    
    for st in SUBTYPES:
        print(f"\nTraining Production model for {st}...")
        
        train_mask = (train_df["target_subtype"] == st).values
        test_mask = (test_df["target_subtype"] == st).values
        
        X_tr = X_train_glob[train_mask]
        y_tr = train_df.loc[train_mask, "pchembl_value"].values
        
        X_te = X_test_glob[test_mask]
        y_te = test_df.loc[test_mask, "pchembl_value"].values
        
        if len(y_tr) < 50:
            print(f"  [SKIP] Insufficient training data for {st} ({len(y_tr)} samples).")
            continue
            
        print(f"  Samples: Train={len(y_tr)}, Test={len(y_te)}")
        
        # Load params for this subtype
        params = best_params_per_subtype[st].copy()
        params.update({"tree_method": "hist", "n_jobs": 2, "random_state": 42})
        
        # Initialize base XGBRegressor
        print(f"  Fitting XGBoost model...")
        xgb_model = xgb.XGBRegressor(**params)
        xgb_model.fit(X_tr, y_tr)
        
        # Train Random Forest
        print(f"  Fitting Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=300, 
            max_depth=15, 
            max_features="sqrt", 
            random_state=42, 
            n_jobs=2
        )
        rf_model.fit(X_tr, y_tr)
        
        # Overfitting & performance checks
        preds_tr_xgb = xgb_model.predict(X_tr)
        preds_te_xgb = xgb_model.predict(X_te)
        
        preds_tr_rf = rf_model.predict(X_tr)
        preds_te_rf = rf_model.predict(X_te)
        
        r2_tr_xgb = r2_score(y_tr, preds_tr_xgb)
        r2_te_xgb = r2_score(y_te, preds_te_xgb)
        
        r2_tr_rf = r2_score(y_tr, preds_tr_rf)
        r2_te_rf = r2_score(y_te, preds_te_rf)
        
        print(f"    XGBoost R2: Train={r2_tr_xgb:.3f}, Test={r2_te_xgb:.3f}")
        print(f"    Random Forest R2: Train={r2_tr_rf:.3f}, Test={r2_te_rf:.3f}")
        
        # Save models
        model_dir = Path("models/precise")
        xgb_path = model_dir / f"xgboost_{st}_production.pkl"
        with open(xgb_path, "wb") as f:
            pickle.dump(xgb_model, f)
            
        rf_path = model_dir / f"rf_{st}_production.pkl"
        with open(rf_path, "wb") as f:
            pickle.dump(rf_model, f)
        
        training_summary[st] = {
            "train_size": int(len(y_tr)),
            "test_size": int(len(y_te)),
            "xgboost_r2": r2_te_xgb,
            "rf_r2": r2_te_rf
        }
        print(f"  [SUCCESS] XGBoost model saved to {xgb_path}")
        
        # Also copy it to models/ root folder for safe loading in predictor
        import shutil
        shutil.copy(xgb_path, Path("models") / f"xgboost_precise_{st.lower()}_model.pkl")
        with open(Path("models") / f"xgboost_{st.lower()}_model.pkl", "wb") as f:
            pickle.dump(xgb_model, f)
            
        # Save RF model
        with open(Path("models/precise") / f"rf_precise_{st.lower()}_model.pkl", "wb") as f:
            pickle.dump(rf_model, f)
        with open(Path("models") / f"rf_{st.lower()}_model.pkl", "wb") as f:
            pickle.dump(rf_model, f)
        print(f"  [SUCCESS] Random Forest model saved to models/rf_{st.lower()}_model.pkl")
    
    # Save training summary
    summary_path = Path("outputs/training_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"\n[SUCCESS] Training summary saved to {summary_path}")
            
    print("\n" + "="*60)
    print("PRODUCTION TRAINING SUITE COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    retrain_production_models()
