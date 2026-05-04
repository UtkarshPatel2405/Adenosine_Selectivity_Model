import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from src.ml_base import (preprocess_data, evaluate_regression,
                         plot_predictions, plot_residuals, save_model)

def train_xgboost(df):
    print("\n" + " " * 80)
    print(" XGBOOST REGRESSION MODEL")
    print(" " * 80)
    
    # Preprocess using the updated scaffold split and selective scaling logic
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(
        df, use_fingerprints=True, use_properties=True, n_bits=2048
    )
    
    print("\n[INFO] Training XGBoost (800 rounds)...")
    model = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=2,
        subsample=0.8,
        tree_method='hist',
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )
    
    # Train with evaluation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    print("[SUCCESS] Training complete!")
    
    # Make predictions
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    test_results = evaluate_regression(y_test, y_pred_test, "XGBoost (Test)")
    
    # Plots
    plot_predictions(y_test, y_pred_test, "XGBoost")
    plot_residuals(y_test, y_pred_test, "XGBoost")
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "="*80)
    print("TOP 15 IMPORTANT FEATURES")
    print("="*80)
    print(importance.head(15))
    
    # Save model
    save_model(model, "XGBoost")
    
    return {
        'model': model,
        'scaler': scaler,
        'test_results': test_results,
        'X_test': X_test,
        'y_test': y_test
    }

if __name__ == "__main__":
    from src.data_loader import load_and_clean
    file_path = "data/raw/AR_all_unique_parents_with_smiles.csv"
    
    # Unpack the tuple and ensure the column name matches the splitter
    df, _lookup = load_and_clean(file_path)
    
    if df is not None:
        if 'canonical_smiles' in df.columns and 'smiles' not in df.columns:
            df = df.rename(columns={'canonical_smiles': 'smiles'})
        train_xgboost(df)