import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from src.ml_base import (preprocess_data_qsar, evaluate_regression,
                         plot_predictions, plot_residuals, save_model)

def train_random_forest(df):
    print("\n" + " "*80)
    print(" RANDOM FOREST REGRESSION MODEL")
    print(" "*80)
    
    # Preprocess
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data_qsar(df, use_fingerprints=True,
    use_properties=True, n_bits=2048)
    
    # Create and train model
    print("\nTraining Random Forest (300 trees)...")
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
    print(" Training complete!")
    
    # Uncertainty quantification
    print("\n" + "="*80)
    print("UNCERTAINTY QUANTIFICATION")
    print("="*80)
    
    # Get predictions from EACH tree (not averaged)
    print("\nCalculating uncertainty from individual tree predictions...")
    
    # Get predictions from each tree
    tree_predictions_train = np.array([tree.predict(X_train) for tree in model.estimators_])
    tree_predictions_test = np.array([tree.predict(X_test) for tree in model.estimators_])
    
    # Calculate uncertainty (standard deviation across trees)
    uncertainty_train = tree_predictions_train.std(axis=0)
    uncertainty_test = tree_predictions_test.std(axis=0)
    
    # Average predictions (normal RF prediction)
    y_pred_train = tree_predictions_train.mean(axis=0)
    y_pred_test = tree_predictions_test.mean(axis=0)
    
    print(f"\n📊 UNCERTAINTY STATISTICS (Test Set):")
    print(f"   Mean uncertainty:   {uncertainty_test.mean():.6f}")
    print(f"   Max uncertainty:    {uncertainty_test.max():.6f}")
    print(f"   Min uncertainty:    {uncertainty_test.min():.6f}")
    print(f"   Std of uncertainty: {uncertainty_test.std():.6f}")
    
    # Evaluate
    train_results = evaluate_regression(y_train, y_pred_train, "Random Forest (Train)")
    test_results = evaluate_regression(y_test, y_pred_test, "Random Forest (Test)")
    
    # Add uncertainty to results
    test_results['uncertainty'] = uncertainty_test
    train_results['uncertainty'] = uncertainty_train
    
    # Plot predictions with uncertainty bounds
    plot_predictions_with_uncertainty(y_test, y_pred_test, uncertainty_test, "Random Forest")
    plot_predictions(y_test, y_pred_test, "Random Forest")
    plot_residuals(y_test, y_pred_test, "Random Forest")
    
    # Plot uncertainty analysis
    plot_uncertainty_analysis(y_test, y_pred_test, uncertainty_test, "Random Forest")
    
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
    plt.barh(top_features['Feature'], top_features['Importance'], color='steelblue', edgecolor='black')
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Random Forest: Top 15 Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('outputs/feature_importance_random_forest.png', dpi=300)
    print("✅ Saved: outputs/feature_importance_random_forest.png")
    plt.close()
    
    # Save model
    save_model(model, "Random Forest")
    
    return {
        'model': model,
        'scaler': scaler,
        'train_results': train_results,
        'test_results': test_results,
        'importance': importance,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'uncertainty_test': uncertainty_test,
        'uncertainty_train': uncertainty_train
    }



# VISUALIZATION FUNCTIONS


def plot_predictions_with_uncertainty(y_test, y_pred, uncertainty, model_name):
    """
    Plot predictions with uncertainty bounds.
    
    VISUALIZATION:
    - Y-axis: Predicted values with error bars
    - X-axis: Sorted actual values
    - Error bars: ±uncertainty (confidence interval)
    """
    
    # Sort by actual values for better visualization
    sort_idx = np.argsort(y_test.values)
    y_test_sorted = y_test.values[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    uncertainty_sorted = uncertainty[sort_idx]
    
    plt.figure(figsize=(14, 6))
    
    # Plot with error bars
    x_range = np.arange(len(y_test_sorted))
    plt.scatter(x_range, y_test_sorted, color='red', s=20, label='Actual', alpha=0.6, zorder=3)
    plt.errorbar(x_range, y_pred_sorted, yerr=uncertainty_sorted, fmt='o', 
                color='steelblue', markersize=4, alpha=0.6, label='Predicted ± Uncertainty', 
                elinewidth=0.5, capsize=2)
    
    plt.xlabel('Sample (sorted by actual value)', fontsize=12)
    plt.ylabel('Potency', fontsize=12)
    plt.title(f'{model_name}: Predictions with Uncertainty Bounds', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/predictions_with_uncertainty_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    print(f"✅ Saved: outputs/predictions_with_uncertainty_{model_name.lower().replace(' ', '_')}.png")
    plt.close()


def plot_uncertainty_analysis(y_test, y_pred, uncertainty, model_name):
    """
    Analyze relationship between prediction error and uncertainty.
    
    LOGIC:
    - Calculate actual error: |y_true - y_pred|
    - Plot error vs uncertainty
    - If correlated: model uncertainty is meaningful
    - If not correlated: uncertainty is just noise
    """
    
    error = np.abs(y_test.values - y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter: Uncertainty vs Error
    axes[0].scatter(uncertainty, error, alpha=0.5, edgecolors='black', s=30)
    
    # Add trend line
    z = np.polyfit(uncertainty, error, 1)
    p = np.poly1d(z)
    axes[0].plot(uncertainty, p(uncertainty), "r--", linewidth=2, label='Trend')
    
    # Calculate correlation
    corr = np.corrcoef(uncertainty, error)[0, 1]
    
    axes[0].set_xlabel('Model Uncertainty (std dev)', fontsize=12)
    axes[0].set_ylabel('Absolute Prediction Error', fontsize=12)
    axes[0].set_title(f'{model_name}: Error vs Uncertainty\n(Correlation: {corr:.4f})', 
                     fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Histogram of uncertainty
    axes[1].hist(uncertainty, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Uncertainty (std dev)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{model_name}: Distribution of Uncertainty', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'outputs/uncertainty_analysis_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    print(f"✅ Saved: outputs/uncertainty_analysis_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    print(f"\n📊 UNCERTAINTY-ERROR CORRELATION:")
    print(f"   Correlation coefficient: {corr:.6f}")
    if corr > 0.5:
        print(f"   ✅ STRONG positive correlation!")
        print(f"      → Model uncertainty PREDICTS actual error")
        print(f"      → Uncertainty estimates are MEANINGFUL")
    elif corr > 0.3:
        print(f"   ⚠️ MODERATE correlation")
        print(f"      → Some relationship exists")
    else:
        print(f"   ❌ WEAK correlation")
        print(f"      → Uncertainty doesn't track error well")


if __name__ == "__main__":
    from src.data_loader import load_data
    
    file_path = r"C:\Users\utkar\Adenosine_Receptor_Lingand\data\raw\AR_all_unique_parents_with_smiles.csv"
    df = load_data(file_path)
    
    if df is not None:
        results = train_random_forest(df)