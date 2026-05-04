import pandas as pd
import os
from src.data_loader import load_and_clean
from src.ml_base import preprocess_data, save_model
from src.ml_xgboost import train_xgboost  # Or your preferred model script

def train_suite(file_path):
   
    df, _lookup = load_and_clean(file_path)
    
    if df is None:
        print("[ERROR] Data loading failed.")
        return

    # Standardize column naming for the feature extractor[cite: 23]
    if 'canonical_smiles' in df.columns and 'smiles' not in df.columns:
        df = df.rename(columns={'canonical_smiles': 'smiles'})

    subtypes = ["A1", "A2A", "A2B", "A3"]
    
    print(f"[INFO] Starting Training Suite for {len(subtypes)} subtypes")

    for st in subtypes:
        print(f"\n" + "="*40)
        print(f"TRAINING SUBTYPE: {st}")
        print("="*40)
        
        # 2. Filter data for the specific receptor[cite: 3]
        subtype_df = df[df['target_subtype'] == st].copy()
        
        if len(subtype_df) < 50:
            print(f"[SKIP] Insufficient data for {st} ({len(subtype_df)} samples).")
            continue
            
        # 3. Train using the corrected pipeline (Scaffold Split + Selective Scaling)[cite: 9, 24]
        # This will internally call your corrected ml_base logic
        results = train_xgboost(subtype_df)
        
        # 4. Save with subtype-specific naming[cite: 5]
        model_name = f"xgboost_{st}"
        save_model(results['model'], model_name)
        print(f"[SUCCESS] Model saved as models/{model_name.lower()}_model.pkl")

if __name__ == "__main__":
    file_path = "data/raw/AR_all_unique_parents_with_smiles.csv"
    train_suite(file_path)