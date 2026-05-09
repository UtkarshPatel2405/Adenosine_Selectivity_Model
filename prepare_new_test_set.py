import pandas as pd
import numpy as np
from pathlib import Path
from src.chem_utils import canonicalize
from rdkit import Chem

def prepare_test_set():
    print("🚀 Starting Test Set Preparation...")

    # 1. Load the existing training data to find all "seen" SMILES
    print("Loading original training dataset...")
    train_df = pd.read_csv("data/raw/AR_all_unique_parents_with_smiles.csv")
    
    # We need to make sure we compare apples to apples, so we canonicalize the training SMILES
    # The data loader handles this, but let's grab it directly
    seen_smiles = set()
    print("Canonicalizing training SMILES for exact matching...")
    for smi in train_df['smiles'].dropna():
        canon = canonicalize(smi)
        if canon:
            seen_smiles.add(canon)
            
    print(f"✅ Found {len(seen_smiles)} unique canonical SMILES in the training set.")

    # 2. Process the new Excel files
    files = {
        "A1": "data/raw/GPCRdb_A1.xlsx",
        "A2A": "data/raw/GPCRdb_A2A.xlsx",
        "A2B": "data/raw/GPCRdb_A2B.xlsx",
        "A3": "data/raw/GPCRdb_A3.xlsx"
    }

    # Dictionary to hold the merged novel molecules
    # Format: {canonical_smiles: {'original_smiles': smi, 'A1': val, 'A2A': val, ...}}
    novel_molecules = {}
    
    total_new_mols = 0
    total_skipped_seen = 0

    for subtype, filepath in files.items():
        if not Path(filepath).exists():
            print(f"⚠️ Warning: {filepath} not found. Skipping.")
            continue
            
        print(f"\nProcessing {subtype} dataset...")
        df = pd.read_excel(filepath)
        print(f"Loaded {len(df)} rows from {subtype}.")
        
        for idx, row in df.iterrows():
            smi = row.get('SMILES')
            p_val = row.get('p-value (-log)')
            
            # Skip missing data
            if pd.isna(smi) or pd.isna(p_val):
                continue
                
            canon = canonicalize(smi)
            if not canon:
                continue
                
            # Filter out molecules we already trained on!
            if canon in seen_smiles:
                total_skipped_seen += 1
                continue
                
            # We found a completely novel molecule
            if canon not in novel_molecules:
                novel_molecules[canon] = {'canonical_smiles': canon, 'original_smiles': smi}
                total_new_mols += 1
                
            # Record the activity for this subtype.
            # If there are duplicates in the same file, we'll just keep the max value
            current_val = novel_molecules[canon].get(subtype, 0)
            novel_molecules[canon][subtype] = max(current_val, float(p_val))

    print(f"\n✅ Processing complete!")
    print(f"🛑 Skipped {total_skipped_seen} occurrences of already-seen molecules.")
    print(f"✨ Found {total_new_mols} completely novel unique molecules!")

    # 3. Convert back to DataFrame and save
    if not novel_molecules:
        print("❌ No novel molecules found! Exiting.")
        return

    # Convert to DataFrame
    novel_df = pd.DataFrame(list(novel_molecules.values()))
    
    # Ensure all subtype columns exist, fill missing with NaN
    for st in ["A1", "A2A", "A2B", "A3"]:
        if st not in novel_df.columns:
            novel_df[st] = np.nan
            
    # Reorder columns nicely
    novel_df = novel_df[['original_smiles', 'canonical_smiles', 'A1', 'A2A', 'A2B', 'A3']]
    
    # Rename to just "smiles" so it works flawlessly with the Batch Predictor App
    novel_df = novel_df.rename(columns={'original_smiles': 'smiles'})
    
    # Create the output directory if needed
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "novel_test_set.csv"
    novel_df.to_csv(out_path, index=False)
    print(f"\n💾 Saved novel test set to: {out_path}")
    print("Ready to run in the Batch Predictor!")

if __name__ == "__main__":
    prepare_test_set()
