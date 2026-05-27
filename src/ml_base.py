import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Import scaffold split logic
from src.scaffold_split import scaffold_split

sns.set_style("whitegrid")

# Create directories 
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Preprocessing
def preprocess_data(df, use_fingerprints=True, use_properties=True, n_bits=2048):
    df = df.reset_index(drop=True)
    print("\n" + "="*80)
    print("DATA PREPROCESSING - MOLECULAR FEATURES UPGRADED PIPELINE")
    print("="*80)
    
    # Standardize column naming
    if 'canonical_smiles' in df.columns and 'smiles' not in df.columns:
        df = df.rename(columns={'canonical_smiles': 'smiles'})
        
    y = df['pchembl_value'].copy()
    valid_smiles = df['smiles'].copy()
    
    # Combine temporarily for scaffold split
    temp_df = pd.DataFrame({'smiles': valid_smiles, 'pchembl_value': y})
    
    # Scaffold split (80-20)
    train_df, test_df = scaffold_split(temp_df, test_size=0.2, random_state=42)
    
    y_train = train_df['pchembl_value'].reset_index(drop=True)
    y_test = test_df['pchembl_value'].reset_index(drop=True)
    
    from src.features import build_feature_matrix
    X_train, X_test, pipeline = build_feature_matrix(train_df, test_df, smiles_col='smiles')
    
    # Generate feature names for DataFrame conversion
    feature_names = [f"Morgan_FP_{i}" for i in range(2048)] + [f"MACCS_{i}" for i in range(167)]
    selected_desc_names = pipeline.feature_filter.feature_names
    feature_names.extend(selected_desc_names)
    
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    print("\n[SUCCESS] Feature processing complete with RDKit, MACCS, and filtered descriptors.")
    return X_train_df, X_test_df, y_train, y_test, pipeline, feature_names

# Evaluation
def evaluate_regression(y_true, y_pred, model_name):
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{'='*80}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*80}")
    print(f"R² Score:  {r2:.6f} (0-1, higher is better)")
    print(f"RMSE:      {rmse:.6f} (lower is better)")
    print(f"MSE:       {mse:.6f}")
    print(f"MAE:       {mae:.6f}")

    interpretation = f"""
    INTERPRETATION:
    - Explains {r2*100:.2f}% of potency variation
    - Average error: ±{rmse:.4f} potency units
    - Typical error: ±{mae:.4f} potency units
    """
    print(interpretation)
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mse': mse,
        'mae': mae,
        'predictions': y_pred
    }

# Plotting functions
def plot_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='black', s=30)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Potency', fontsize=12)
    plt.ylabel('Predicted Potency', fontsize=12)
    plt.title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    filename = f'outputs/pred_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300)
    print(f"[SUCCESS] Saved: {filename}")
    plt.close()

def plot_residuals(y_test, y_pred, model_name):
    residuals = y_test.values - y_pred
    plt.figure(figsize=(12, 6))

    # Residual plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5, edgecolors='black', s=30)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Values', fontsize=11)
    plt.ylabel('Residuals', fontsize=11)
    plt.title('Residual Plot', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)

    # Histogram of residuals
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel('Residuals', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Distribution of Residuals', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = f'outputs/residuals_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300)
    print(f"[SUCCESS] Saved: {filename}")
    plt.close()

# Save/Load functions
def save_model(model, model_name):
    """Save trained model to disk."""
    filepath = f'models/{model_name.lower().replace(" ", "_")}_model.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"[SUCCESS] Model saved: {filepath}")

def load_model(model_name):
    """Load trained model from disk."""
    filepath = f'models/{model_name.lower().replace(" ", "_")}_model.pkl'
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"[SUCCESS] Model loaded: {filepath}")
        return model
    except FileNotFoundError:
        print(f"[ERROR] Model not found: {filepath}")
        return None