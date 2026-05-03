import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from src.ml_base import (preprocess_data, evaluate_regression,
                         plot_predictions, plot_residuals, save_model)

def train_xgboost(df):
    print("\n" + " "*80)
    print(" XGBOOST REGRESSION MODEL")
    print(" "*80)
    
    # Preprocess
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df)
    
    # Create and train model
    print("\nTraining XGBoost (200 rounds)...")
    model = xgb.XGBRegressor(
        n_estimators=200,          # Boosting rounds
        learning_rate=0.1,         # Learning rate
        max_depth=6,               # Tree depth
        min_child_weight=1,
        subsample=0.8,             # 80% of data per round
        colsample_bytree=0.8,      # 80% of features per round
        random_state=42,
        n_jobs=-1
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    print(" Training complete!")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_results = evaluate_regression(y_train, y_pred_train, "XGBoost (Train)")
    test_results = evaluate_regression(y_test, y_pred_test, "XGBoost (Test)")
    
    # Plot
    plot_predictions(y_test, y_pred_test, "XGBoost")
    plot_residuals(y_test, y_pred_test, "XGBoost")
    
    # Feature importance
    print("\n" + "="*80)
    print("TOP 15 IMPORTANT FEATURES")
    print("="*80)
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(importance.head(15))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = importance.head(15)
    plt.barh(top_features['Feature'], top_features['Importance'], color='coral', edgecolor='black')
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('XGBoost: Top 15 Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('outputs/feature_importance_xgboost.png', dpi=300)
    print(" Saved: outputs/feature_importance_xgboost.png")
    plt.close()
    
    # Save model
    save_model(model, "XGBoost")
    
    return {
        'model': model,
        'scaler': scaler,
        'train_results': train_results,
        'test_results': test_results,
        'importance': importance,
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
        results = train_xgboost(df)