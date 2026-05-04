import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from src.ml_base import (preprocess_data, evaluate_regression, 
                         plot_predictions, plot_residuals, save_model)

def train_linear_regression(df):
    print("\n" + " " * 80)
    print(" LINEAR REGRESSION MODEL")
    print(" " * 80)
    
    # Preprocess using the updated scaffold split logic
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(
        df, use_fingerprints=True, use_properties=True, n_bits=2048
    )
    
    print("\n[INFO] Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[SUCCESS] Training complete!")
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    evaluate_regression(y_train, y_pred_train, "Linear Regression (Train)")
    test_results = evaluate_regression(y_test, y_pred_test, "Linear Regression (Test)")
    
    plot_predictions(y_test, y_pred_test, "Linear Regression")
    plot_residuals(y_test, y_pred_test, "Linear Regression")
    
    coefficients = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("\n" + "="*80)
    print("TOP 10 FEATURE COEFFICIENTS")
    print("="*80)
    print(coefficients.head(10))
    
    save_model(model, "Linear Regression")
    
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
    
    # Correctly unpack tuple and rename column for feature extraction
    df, _lookup = load_and_clean(file_path)
    if df is not None:
        if 'canonical_smiles' in df.columns and 'smiles' not in df.columns:
            df = df.rename(columns={'canonical_smiles': 'smiles'})
        train_linear_regression(df)