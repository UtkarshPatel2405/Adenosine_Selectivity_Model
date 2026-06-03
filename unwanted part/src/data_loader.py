import json
import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem


SUBTYPE_MAP = {
    "A1R": "A1",
    "A2AR": "A2A",
    "A2BR": "A2B",
    "A3R": "A3",
    "A1": "A1",
    "A2A": "A2A",
    "A2B": "A2B",
    "A3": "A3",
}


def _canonicalize_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_and_clean(
    file_path: str,
    save_lookup_path: str = "data/processed/db_lookup.json",
    mode: str = "precise",
    target_role: str = "all",
    target_endpoint: str = "all",
    include_decoys: bool = False,
):
    print(f"\n[INFO] Running precise filtering on {file_path} (Role={target_role}, Endpoint={target_endpoint}, Decoy Ingestion={include_decoys})")
    df = pd.read_csv(file_path)

    required = {
        "smiles",
        "pchembl_value",
        "standard_type",
        "TAG",
        "standard_relation",
        "standard_units",
        "assay_type",
        "confidence_score",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    # 1. Standardize columns
    df["standard_type"] = df["standard_type"].astype(str).str.upper().str.strip()
    df["standard_relation"] = df["standard_relation"].astype(str).str.strip()
    df["assay_type"] = df["assay_type"].astype(str).str.upper().str.strip()
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")

    # 2. Precise scientific filters:
    # - Exact relationships only to avoid bound-based pollution
    # - High confidence target assignment (confidence_score >= 6)
    # - Direct binding (B) or functional (F) assays
    # - Supported standard types and role filtering
    # - Valid pChEMBL value
    initial_count = len(df)
    
    # Base filter
    df = df[
        (df["standard_relation"] == "=") &
        (df["confidence_score"] >= 6) &
        (df["assay_type"].isin({"B", "F"})) &
        (df["pchembl_value"].notna())
    ].copy()
    
    # Endpoint filtering
    if target_endpoint != "all":
        if target_endpoint.upper() in {"KI", "KD"}:
            df = df[df["standard_type"].isin({"KI", "KD"})].copy()
        elif target_endpoint.upper() in {"IC50", "EC50"}:
            df = df[df["standard_type"].isin({"IC50", "EC50"})].copy()
    else:
        df = df[df["standard_type"].isin({"KI", "KD", "IC50", "EC50"})].copy()
        
    # Role filtering
    if target_role != "all":
        if "role" in df.columns:
            df = df[df["role"].astype(str).str.lower() == target_role.lower()].copy()

    # Map subtypes
    df["TAG"] = df["TAG"].astype(str).str.strip()
    df["target_subtype"] = df["TAG"].map(SUBTYPE_MAP)
    df = df[df["target_subtype"].notna()].copy()

    # Canonicalize SMILES
    df["canonical_smiles"] = df["smiles"].apply(_canonicalize_smiles)
    df = df[df["canonical_smiles"].notna()].copy()

    post_filter_count = len(df)
    print(f"[INFO] Scientific filters: {initial_count} raw rows -> {post_filter_count} high-quality rows.")

    # 3. Median-based priority deduplication
    # Group by canonical_smiles and target_subtype
    grouped = df.groupby(["canonical_smiles", "target_subtype"])
    
    deduped_rows = []
    shifts = []
    
    for (smi, subtype), group in grouped:
        if len(group) == 1:
            deduped_rows.append(group.iloc[0])
            continue
        
        # Priority: Ki/Kd first, then IC50/EC50
        binding_group = group[group["standard_type"].isin({"KI", "KD"})]
        if not binding_group.empty:
            target_group = binding_group
        else:
            target_group = group
            
        median_pchembl = target_group["pchembl_value"].median()
        
        # Calculate shift metrics for logging
        orig_max = group["pchembl_value"].max()
        shift = abs(orig_max - median_pchembl)
        shifts.append(shift)
        
        # Keep one representative row, update its pchembl_value to the median
        rep_row = target_group.iloc[0].copy()
        rep_row["pchembl_value"] = median_pchembl
        deduped_rows.append(rep_row)

    df_deduped = pd.DataFrame(deduped_rows).reset_index(drop=True)
    final_count = len(df_deduped)
    
    mean_shift = sum(shifts) / len(shifts) if shifts else 0.0
    max_shift = max(shifts) if shifts else 0.0
    
    print(f"[INFO] Deduplication collapsed {post_filter_count} rows -> {final_count} unique (SMILES, Subtype) pairs.")
    print(f"[INFO] Collapse stats: mean pChEMBL shift = {mean_shift:.3f}, max shift = {max_shift:.3f}")

    # Build database lookup dictionary
    lookup = {}
    for smi, subdf in df_deduped.groupby("canonical_smiles"):
        lookup[smi] = {
            row["target_subtype"]: float(row["pchembl_value"])
            for _, row in subdf.iterrows()
        }

    # Inject high-quality mutual decoys if requested
    if include_decoys:
        print("[INFO] Programmatic mutual decoy generation triggered...")
        all_smiles = df_deduped["canonical_smiles"].unique()
        decoy_rows = []
        SUBTYPES = ["A1", "A2A", "A2B", "A3"]
        for smiles in all_smiles:
            subtype_vals = lookup.get(smiles, {})
            # Check if active on any other subtype (pChEMBL >= 6.5)
            actives = [sub for sub, val in subtype_vals.items() if val >= 6.5]
            if actives:
                # Add a decoy row for any subtype NOT present (representing non-binder off-targets)
                for sub in SUBTYPES:
                    if sub not in subtype_vals:
                        decoy_rows.append({
                            "TAG": f"{sub}R",
                            "canonical_smiles": smiles,
                            "pchembl_value": 3.0,  # Decoy inactive affinity
                            "target_subtype": sub,
                            "standard_type": "DECOY",
                        })
                        
        # Ingest structural P2Y decoys recommended by professor to define GPCR class boundary
        p2y_path = Path("data/processed/p2y_decoys.csv")
        if p2y_path.exists():
            print(f"[INFO] Ingesting structural P2Y decoys from {p2y_path}...")
            p2y_df = pd.read_csv(p2y_path)
            p2y_count = 0
            for _, row in p2y_df.iterrows():
                smiles = row["canonical_smiles"]
                for sub in SUBTYPES:
                    decoy_rows.append({
                        "TAG": f"{sub}R",
                        "canonical_smiles": smiles,
                        "pchembl_value": 3.0,  # Decoy inactive affinity
                        "target_subtype": sub,
                        "standard_type": "DECOY_P2Y"
                    })
                    p2y_count += 1
            print(f"[SUCCESS] Ingested {p2y_count} structural P2Y non-binder controls.")
            
        if decoy_rows:
            decoy_df = pd.DataFrame(decoy_rows)
            df_deduped = pd.concat([df_deduped, decoy_df], ignore_index=True)
            print(f"[SUCCESS] Ingested {len(decoy_df)} high-quality mutual decoy (non-binder) controls.")
            
            # Rebuild lookup to include the newly generated decoys!
            lookup = {}
            for smi, subdf in df_deduped.groupby("canonical_smiles"):
                lookup[smi] = {
                    row["target_subtype"]: float(row["pchembl_value"])
                    for _, row in subdf.iterrows()
                }

    # Save lookup to disk
    Path(save_lookup_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_lookup_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, sort_keys=True)

    keep_cols = ["TAG", "canonical_smiles", "pchembl_value", "target_subtype", "standard_type"]
    df_deduped = df_deduped[keep_cols].copy()

    return df_deduped, lookup

