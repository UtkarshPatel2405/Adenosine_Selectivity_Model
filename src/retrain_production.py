import os
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

from src.data_loader import load_and_clean
from src.features import build_feature_matrix, build_features
from src.conformal import train_conformal_model
from src.scaffold_split import scaffold_split

SUBTYPES = ["A1", "A2A", "A2B", "A3"]

# Robust fallback hyperparameters if nested CV hasn't been run yet
DEFAULT_PARAMS = {
    "A1": {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A2A": {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A2B": {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A3": {"n_estimators": 800, "max_depth": 7, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2}
}

def retrain_production_models(data_path: str = "data/raw/AR_all_unique_parents_with_smiles.csv"):
    print("\n" + "="*60)
    print("PRODUCTION TRAINING WITH MAPIE CONFORMAL WRAPPING")
    print("="*60)
    
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
        
    # 2. Load clean precise data
    df, _ = load_and_clean(data_path, mode="precise")
    df = df.rename(columns={"canonical_smiles": "smiles"})
    
    # 3. Create global scaffold split (80-20) to have test evaluations
    train_df, test_df = scaffold_split(df, test_size=0.2, random_state=42, smiles_col="smiles")
    
    # 4. Build features globally on precise dataset
    X_train_glob, X_test_glob, pipeline = build_feature_matrix(train_df, test_df, smiles_col="smiles")
    
    # Save precise scalers
    os.makedirs("models/precise", exist_ok=True)
    with open("models/precise/scaler_precise.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("[SUCCESS] Saved production scaler pipelines to models/scaler.pkl and models/precise/")
    
    # 5. Train each subtype model
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
        params.update({"tree_method": "hist", "n_jobs": -1, "random_state": 42})
        
        # Initialize base XGBRegressor
        base_xgb = xgb.XGBRegressor(**params)
        
        # Wrap with conformal wrapper
        # 5-fold CV internally to save out-of-fold residuals for conformal intervals
        print(f"  Fitting conformal model CrossConformalRegressor(XGBoost, cv=5, method='plus')...")
        conformal_model = train_conformal_model(base_xgb, X_tr, y_tr, cv=5)
        
        # Save model
        model_name = f"xgboost_precise_{st.lower()}_model.pkl"
        model_path = Path("models/precise") / model_name
        with open(model_path, "wb") as f:
            pickle.dump(conformal_model, f)
        print(f"  [SUCCESS] Conformal model saved to {model_path}")
        
        # Also copy it to models/ root folder for safe loading in predictor
        with open(Path("models") / f"xgboost_{st.lower()}_model.pkl", "wb") as f:
            pickle.dump(conformal_model, f)
        with open(Path("models") / f"xgboost_precise_{st.lower()}_model.pkl", "wb") as f:
            pickle.dump(conformal_model, f)
            
    print("\n" + "="*60)
    print("PRODUCTION TRAINING SUITE COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    retrain_production_models()
