import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

sns.set_style("whitegrid")

# create directories 
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Preprocessing
def preprocess_data_qsar(df, use_fingerprints=True, use_properties=True, n_bits=2048):
    from src.molecular_features import extract_features_from_smiles
    
    print("\n" + "="*80)
    print("DATA PREPROCESSING - TRUE QSAR MODEL (NO DATA LEAKAGE)")
    print("="*80)
    
    print(f"\n⚠️ REMOVING DATA LEAKAGE:")
    print(f"   ❌ EXCLUDING: standard_value (data leakage!)")
    print(f"   ❌ EXCLUDING: confidence_score")
    print(f"   ❌ EXCLUDING: n_targets_hit")
    print(f"   ✅ INCLUDING: Morgan Fingerprints (chemical structure)")
    print(f"   ✅ INCLUDING: Physicochemical Properties (molecular descriptors)")
    
    # Extract features from SMILES
    X, valid_indices = extract_features_from_smiles(
        df, 
        use_fingerprints=use_fingerprints,
        use_properties=use_properties,
        n_bits=n_bits
    )
    
    if X is None:
        print("❌ Failed to extract features!")
        return None
    
    # Get target values for valid molecules
    y = df.loc[valid_indices, 'pchembl_value'].copy()
    
    print(f"\n📊 DATASET INFO:")
    print(f"   Total molecules: {len(df)}")
    print(f"   Valid molecules: {len(valid_indices)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target: {y.shape[0]} values")
    print(f"   Target range: {y.min():.2f} to {y.max():.2f}")
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print(f"\n✅ Scaling complete!")
    print(f"Scaled range: {X_train_scaled.values.min():.2f} to {X_train_scaled.values.max():.2f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

#Evaluation
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

#Ploting functions
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
    plt.savefig(f'outputs/pred_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    print(f"✅ Saved: outputs/pred_{model_name.lower().replace(' ', '_')}.png")
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
    plt.savefig(f'outputs/residuals_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    print(f"✅ Saved: outputs/residuals_{model_name.lower().replace(' ', '_')}.png")
    plt.close()


# save model

def save_model(model, model_name):
    """Save trained model to disk."""
    
    filepath = f'models/{model_name.lower().replace(" ", "_")}_model.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved: {filepath}")


def load_model(model_name):
    """Load trained model from disk."""
    
    filepath = f'models/{model_name.lower().replace(" ", "_")}_model.pkl'
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded: {filepath}")
        return model
    except:
        print(f"❌ Model not found: {filepath}")
        return None
