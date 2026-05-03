import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.ml_linear_regression import train_linear_regression
from src.ml_random_forest import train_random_forest
from src.ml_xgboost import train_xgboost
from src.ml_neural_network import train_neural_network

def train_all_models(df):
    print("\n\n" + "="*80)
    print("TRAINING ALL 4 MODELS")
    print("="*80)
    
    # Train each model
    print("\n\n[1/4] Training Linear Regression...")
    lr_results = train_linear_regression(df)
    
    print("\n\n[2/4] Training Random Forest...")
    rf_results = train_random_forest(df)
    
    print("\n\n[3/4] Training XGBoost...")
    xgb_results = train_xgboost(df)
    
    print("\n\n[4/4] Training Neural Network...")
    nn_results = train_neural_network(df)
    
    return {
        'Linear Regression': lr_results,
        'Random Forest': rf_results,
        'XGBoost': xgb_results,
        'Neural Network': nn_results
    }


def create_ensemble(all_results):
    """Combine predictions from all models."""
    
    print("\n\n" + "="*80)
    print("CREATING ENSEMBLE MODEL")
    print("="*80)
    
    # Get predictions from each model
    lr_pred = all_results['Linear Regression']['test_results']['predictions']
    rf_pred = all_results['Random Forest']['test_results']['predictions']
    xgb_pred = all_results['XGBoost']['test_results']['predictions']
    nn_pred = all_results['Neural Network']['test_results']['predictions'].flatten()
    
    # Average ensemble
    ensemble_pred = (lr_pred + rf_pred + xgb_pred + nn_pred) / 4
    
    # Get actual values
    y_test = all_results['Linear Regression']['y_test']
    
    # Calculate metrics
    from src.ml_base import evaluate_regression
    ensemble_results = evaluate_regression(y_test, ensemble_pred, "ENSEMBLE (All 4 Models)")
    
    # Plot comparison
    plot_model_comparison(all_results, ensemble_pred, y_test)
    
    return ensemble_results


def plot_model_comparison(all_results, ensemble_pred, y_test):
    """Compare all models and ensemble."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    models = ['Linear Regression', 'Random Forest', 'XGBoost', 'Neural Network']
    
    for idx, model_name in enumerate(models):
        y_pred = all_results[model_name]['test_results']['predictions']
        
        axes[idx].scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors='black')
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                      'r--', lw=2, label='Perfect')
        axes[idx].set_xlabel('Actual Potency')
        axes[idx].set_ylabel('Predicted Potency')
        
        r2 = all_results[model_name]['test_results']['r2']
        rmse = all_results[model_name]['test_results']['rmse']
        
        axes[idx].set_title(f'{model_name}\nR²={r2:.4f}, RMSE={rmse:.4f}',
                           fontweight='bold', fontsize=12)
        axes[idx].grid(alpha=0.3)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/model_comparison_all.png', dpi=300)
    print(" Saved: outputs/model_comparison_all.png")
    plt.close()
    
    # Performance metrics comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = ['Linear\nRegression', 'Random\nForest', 'XGBoost', 'Neural\nNetwork']
    r2_scores = [
        all_results['Linear Regression']['test_results']['r2'],
        all_results['Random Forest']['test_results']['r2'],
        all_results['XGBoost']['test_results']['r2'],
        all_results['Neural Network']['test_results']['r2']
    ]
    
    colors = ['steelblue', 'coral', 'lightgreen', 'gold']
    bars = ax.bar(model_names, r2_scores, color=colors, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/model_performance_comparison.png', dpi=300)
    print(" Saved: outputs/model_performance_comparison.png")
    plt.close()