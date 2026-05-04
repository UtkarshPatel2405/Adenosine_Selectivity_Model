import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from src.ml_base import (preprocess_data, evaluate_regression,
                         plot_predictions, plot_residuals, save_model)

def train_random_forest(df):
    print("\n" + " " * 80)
    print(" RANDOM FOREST REGRESSION MODEL")
    print(" " * 80)
    
    # Preprocess using the updated scaffold split and selective scaling logic
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(
        df, use_fingerprints=True, use_properties=True, n_bits=2048
    )
    
    print("\n[INFO] Training Random Forest (300 trees)...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=4,
        min_samples_split=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("[SUCCESS] Training complete!")
    
    # Uncertainty quantification
    print("\n" + "="*80)
    print("UNCERTAINTY QUANTIFICATION")
    print("="*80)
    
    # Get predictions from each tree for uncertainty estimation
    tree_preds_test = np.array([tree.predict(X_test) for tree in model.estimators_])
    uncertainty_test = tree_preds_test.std(axis=0)
    y_pred_test = tree_preds_test.mean(axis=0)
    
    print(f"\n[INFO] UNCERTAINTY STATISTICS (Test Set):")
    print(f"   Mean uncertainty:   {uncertainty_test.mean():.6f}")
    print(f"   Max uncertainty:    {uncertainty_test.max():.6f}")
    
    # Evaluate
    test_results = evaluate_regression(y_test, y_pred_test, "Random Forest (Test)")
    
    # Standard Plots
    plot_predictions(y_test, y_pred_test, "Random Forest")
    plot_residuals(y_test, y_pred_test, "Random Forest")
    
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
    save_model(model, "Random Forest")
    
    return {
        'model': model,
        'scaler': scaler,
        'test_results': test_results,
        'X_test': X_test,
        'y_test': y_test,
        'uncertainty_test': uncertainty_test
    }

if __name__ == "__main__":
    from src.data_loader import load_and_clean
    file_path = "data/raw/AR_all_unique_parents_with_smiles.csv"
    
    # Unpack the tuple and ensure the column name matches the splitter
    df, _lookup = load_and_clean(file_path)
    
    if df is not None:
        if 'canonical_smiles' in df.columns and 'smiles' not in df.columns:
            df = df.rename(columns={'canonical_smiles': 'smiles'})
        train_random_forest(df)