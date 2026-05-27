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
):
    print(f"\n[INFO] Running precise filtering on {file_path} (Requested mode: {mode} -> Unified Precise Filter)")
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
    # - Supported standard types
    # - Valid pChEMBL value
    initial_count = len(df)
    
    df = df[
        (df["standard_relation"] == "=") &
        (df["confidence_score"] >= 6) &
        (df["assay_type"].isin({"B", "F"})) &
        (df["standard_type"].isin({"KI", "KD", "IC50", "EC50"})) &
        (df["pchembl_value"].notna())
    ].copy()

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

    # Save lookup to disk
    Path(save_lookup_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_lookup_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, sort_keys=True)

    keep_cols = ["TAG", "canonical_smiles", "pchembl_value", "target_subtype", "standard_type"]
    df_deduped = df_deduped[keep_cols].copy()

    return df_deduped, lookup

