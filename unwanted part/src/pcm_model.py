# src/pcm_model.py
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

from src.features import build_feature_matrix, build_features
from src.conformal import train_conformal_model
from src.scaffold_split import scaffold_split

SUBTYPES = ["A1", "A2A", "A2B", "A3"]

def build_pcm_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, smiles_col: str = "smiles"):
    """
    Featurize SMILES and concatenate with target subtype one-hot encodings.
    Returns:
        X_train_pcm: numpy array of shape (N_train, D_ligand + 4)
        y_train_pcm: numpy array of shape (N_train,)
        X_test_pcm: numpy array of shape (N_test, D_ligand + 4)
        y_test_pcm: numpy array of shape (N_test,)
        pipeline: FeaturePipeline instance for ligand features
    """
    print("[INFO] Building unified Proteochemometric (PCM) feature dataset...")
    
    # 1. Build molecular descriptors for all compounds globally
    # To keep features.py pipelines clean, we build a global feature matrix first
    X_train_lig, X_test_lig, pipeline = build_feature_matrix(train_df, test_df, smiles_col=smiles_col, save_to_disk=False)
    
    # 2. Reconstruct one-hot encoding for the 4 subtypes
    # Subtypes: A1, A2A, A2B, A3
    def get_one_hot(subtype_series):
        # Create dictionary mapping subtype to index
        idx_map = {st: i for i, st in enumerate(SUBTYPES)}
        encoded = np.zeros((len(subtype_series), len(SUBTYPES)), dtype=np.float32)
        for i, sub in enumerate(subtype_series.tolist()):
            if sub in idx_map:
                encoded[i, idx_map[sub]] = 1.0
        return encoded

    train_oh = get_one_hot(train_df["target_subtype"])
    test_oh = get_one_hot(test_df["target_subtype"])
    
    # 3. Concatenate ligand features with subtype one-hot vectors
    X_train_pcm = np.hstack([X_train_lig, train_oh]).astype(np.float32)
    X_test_pcm = np.hstack([X_test_lig, test_oh]).astype(np.float32)
    
    y_train_pcm = train_df["pchembl_value"].values.astype(np.float32)
    y_test_pcm = test_df["pchembl_value"].values.astype(np.float32)
    
    print(f"[SUCCESS] PCM Dataset constructed:")
    print(f"  Train: Shape={X_train_pcm.shape}, Targets={y_train_pcm.shape}")
    print(f"  Test:  Shape={X_test_pcm.shape}, Targets={y_test_pcm.shape}")
    
    return X_train_pcm, y_train_pcm, X_test_pcm, y_test_pcm, pipeline

def train_pcm_production_model(train_df: pd.DataFrame, test_df: pd.DataFrame, params=None):
    """
    Train production-ready conformal wrapped PCM model.
    """
    if params is None:
        params = {
            "n_estimators": 800,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 2,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42
        }
    
    X_tr, y_tr, X_te, y_te, pipeline = build_pcm_dataset(train_df, test_df, smiles_col="smiles")
    
    # Fit base XGBoost
    base_xgb = xgb.XGBRegressor(**params)
    
    print("[INFO] Training Conformal Mapie Wrapper for PCM...")
    conformal_pcm = train_conformal_model(base_xgb, X_tr, y_tr, cv=5)
    
    # Evaluate
    preds, y_pis = conformal_pcm.predict_interval(X_te)
    r2 = float(r2_score(y_te, preds))
    mae = float(mean_absolute_error(y_te, preds))
    rmse = float(np.sqrt(np.mean((y_te - preds) ** 2)))
    
    print(f"\n[PCM EVALUATION] scaffold test set metrics:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  RMSE:     {rmse:.4f}")
    
    # Save the PCM models and pipeline
    out_dir = Path("models/pcm")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "xgboost_pcm_model.pkl", "wb") as f:
        pickle.dump(conformal_pcm, f)
    with open(out_dir / "scaler_pcm.pkl", "wb") as f:
        pickle.dump(pipeline, f)
        
    print(f"[SUCCESS] PCM models successfully written to {out_dir}")
    
    return conformal_pcm, pipeline, {"r2": r2, "mae": mae, "rmse": rmse}
