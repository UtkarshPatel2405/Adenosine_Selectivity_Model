import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from src.ml_base import (preprocess_data_qsar, evaluate_regression, 
                         plot_predictions, plot_residuals, save_model)

def train_linear_regression(df):
    print("\n" + " "*80)
    print(" LINEAR REGRESSION MODEL")
    print(" "*80)
    
    # Preprocess
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data_qsar(df,  use_fingerprints=True,
    use_properties=True, n_bits=2048)
    
    # Create and train model
    print("\nTraining Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_results = evaluate_regression(y_train, y_pred_train, "Linear Regression (Train)")
    test_results = evaluate_regression(y_test, y_pred_test, "Linear Regression (Test)")
    
    # Plot
    plot_predictions(y_test, y_pred_test, "Linear Regression")
    plot_residuals(y_test, y_pred_test, "Linear Regression")
    
    # Feature importance
    print("\n" + "="*80)
    print("TOP 10 FEATURE COEFFICIENTS")
    print("="*80)
    coefficients = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print(coefficients.head(10))
    
    # Save model
    save_model(model, "Linear Regression")
    
    return {
        'model': model,
        'scaler': scaler,
        'train_results': train_results,
        'test_results': test_results,
        'coefficients': coefficients,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


if __name__ == "__main__":
    from src.data_loader import load_data
    
    file_path = r"C:\Users\utkar\Adenosine_Receptor_Lingand\data\raw\AR_all_unique_parents_with_smiles.csv"
    df = load_data(file_path)
    
    if df is not None:
        results = train_linear_regression(df)
