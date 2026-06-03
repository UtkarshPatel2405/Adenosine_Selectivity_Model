import os
import time
import requests
import pandas as pd
from rdkit import Chem

# Target mappings
P2Y_TARGETS = {
    "P2Y1": "CHEMBL4315",
    "P2Y2": "CHEMBL4398",
    "P2Y12": "CHEMBL2001"
}

def _canonicalize_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def fetch_p2y_compounds():
    print("="*70)
    print("FETCHING P2Y ANTAGONISTS FROM CHEMBL AS STRUCTURAL DECOYS")
    print("="*70)
    
    # 1. Load active AR SMILES to filter out any overlapping binders
    ar_data_path = "data/raw/AR_all_unique_parents_with_smiles.csv"
    ar_actives = set()
    if os.path.exists(ar_data_path):
        ar_df = pd.read_csv(ar_data_path)
        # Use smiles column (since we standardized it) or canonical_smiles
        smiles_col = "smiles" if "smiles" in ar_df.columns else "canonical_smiles"
        for smi in ar_df[smiles_col].dropna():
            canon = _canonicalize_smiles(smi)
            if canon:
                ar_actives.add(canon)
        print(f"[INFO] Loaded {len(ar_actives)} active Adenosine Receptor ligands from ChEMBL.")
    else:
        print("[WARNING] Adenosine raw dataset not found. Overlap filtering will be skipped.")
        
    p2y_compounds = []
    
    for name, tid in P2Y_TARGETS.items():
        print(f"\nQuerying ChEMBL activities for human {name} (Target ID: {tid})...")
        url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        params = {
            "target_chembl_id": tid,
            "pchembl_value__gte": 6.0,
            "standard_type__in": "IC50,Ki,Kd,EC50",
            "limit": 1000
        }
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            activities = data.get("activities", [])
            print(f"  Retrieved {len(activities)} activity records.")
            
            count_added = 0
            for act in activities:
                smi = act.get("canonical_smiles")
                cid = act.get("molecule_chembl_id")
                pchembl = act.get("pchembl_value")
                
                if not smi or not cid:
                    continue
                    
                canon = _canonicalize_smiles(smi)
                if not canon:
                    continue
                    
                # Strict check: Must not overlap with AR actives
                if canon in ar_actives:
                    continue
                    
                p2y_compounds.append({
                    "parent_molecule_chembl_id": cid,
                    "canonical_smiles": canon,
                    "pchembl_value": 3.0,  # Assigned decoy/inactive affinity for AR subtypes
                    "source_target": name,
                    "source_pchembl": pchembl,
                    "standard_type": "DECOY_P2Y"
                })
                count_added += 1
                
            print(f"  Added {count_added} unique non-overlapping P2Y ligands.")
        except Exception as e:
            print(f"  [ERROR] Failed to query {name}: {e}")
            
    if not p2y_compounds:
        print("\n[ERROR] No P2Y antagonists retrieved!")
        return
        
    df_p2y = pd.DataFrame(p2y_compounds)
    # Deduplicate by canonical_smiles
    df_p2y = df_p2y.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/p2y_decoys.csv"
    df_p2y.to_csv(out_path, index=False)
    print("\n" + "="*70)
    print(f"[SUCCESS] Wrote {len(df_p2y)} high-quality structural P2Y decoys to {out_path}")
    print("="*70)

if __name__ == "__main__":
    fetch_p2y_compounds()
