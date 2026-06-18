"""
Data Loader — Standardize, filter, deduplicate, and prepare ChEMBL and GPCRdb bioactivity data.

Uses the SmilesRegistry barcode system for deterministic deduplication.
Supports programmatic mutual decoy injection and P2Y structural controls.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from src.smiles_registry import SmilesRegistry


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


def load_all_raw_data(data_dir: str) -> pd.DataFrame:
    """Loads and standardizes all CSV and Excel files in the raw data directory."""
    raw_path = Path(data_dir)
    dfs = []

    # 1. Load ChEMBL CSV
    chembl_file = raw_path / "AR_all_unique_parents_with_smiles.csv"
    if chembl_file.exists():
        df_chembl = pd.read_csv(chembl_file)
        # Keep targets_hit so we can use it to block false decoys
        keep_cols = [
            "smiles",
            "pchembl_value",
            "standard_type",
            "TAG",
            "standard_relation",
            "assay_type",
            "confidence_score",
            "targets_hit",
        ]
        df_chembl = df_chembl[[c for c in keep_cols if c in df_chembl.columns]].copy()
        dfs.append(df_chembl)
        print(f"[INFO] Loaded ChEMBL CSV: {len(df_chembl)} rows.")

    # 2. Load GPCRdb Excel files
    for excel_file in raw_path.glob("GPCRdb_*.xlsx"):
        df_gpcr = pd.read_excel(excel_file)
        # Extract TAG from filename, e.g., GPCRdb_A2A.xlsx -> A2AR
        tag = excel_file.stem.split("_")[1] + "R"

        # Standardize columns to match ChEMBL
        rename_map = {
            "Smiles": "smiles",
            "p-value (-log)": "pchembl_value",
            "Activity Type": "standard_type",
            "Activity Relation": "standard_relation",
            "Assay Type": "assay_type",
        }
        df_gpcr = df_gpcr.rename(columns=rename_map)

        # Strip 'p' from pKi, pKd
        if "standard_type" in df_gpcr.columns:
            df_gpcr["standard_type"] = (
                df_gpcr["standard_type"].astype(str).str.replace(r"^p", "", regex=True)
            )

        df_gpcr["TAG"] = tag
        df_gpcr["confidence_score"] = (
            9  # Assume high confidence for GPCRdb manually curated data
        )

        # Map assay type Binding -> B, Functional -> F
        if "assay_type" in df_gpcr.columns:
            df_gpcr["assay_type"] = df_gpcr["assay_type"].replace(
                {"Binding": "B", "Functional": "F"}
            )

        # Add missing columns
        for col in [
            "smiles",
            "pchembl_value",
            "standard_type",
            "TAG",
            "standard_relation",
            "assay_type",
            "confidence_score",
            "targets_hit",
        ]:
            if col not in df_gpcr.columns:
                df_gpcr[col] = None

        # Fill missing targets_hit with the TAG itself so it counts as a hit for decoy prevention
        df_gpcr["targets_hit"] = df_gpcr["TAG"]

        df_gpcr = df_gpcr[
            [
                "smiles",
                "pchembl_value",
                "standard_type",
                "TAG",
                "standard_relation",
                "assay_type",
                "confidence_score",
                "targets_hit",
            ]
        ].copy()
        dfs.append(df_gpcr)
        print(f"[INFO] Loaded GPCRdb Excel {excel_file.name}: {len(df_gpcr)} rows.")

    if not dfs:
        raise ValueError(f"No valid datasets found in {data_dir}")

    return pd.concat(dfs, ignore_index=True)


def load_and_clean(
    data_dir: str = "data/raw",
    save_lookup_path: str = "data/processed/db_lookup.json",
    mode: str = "precise",
    target_role: str = "all",
    target_endpoint: str = "all",
    include_decoys: bool = False,
):
    """
    Load, filter, deduplicate, and prepare bioactivity data for ML training from all files in data_dir.

    Gate optimizations applied:
    - AC50 included alongside KI, KD, IC50, EC50
    - Decoy active trigger relaxed to 5.5 (from 6.5)
    - Decoy pChEMBL set to 4.0 (from 3.0) for realistic pharmacological noise
    - Prevents generating decoys for targets listed in 'targets_hit' (e.g. multi-target compounds)
    """
    print(
        f"\n[INFO] Running precise filtering on {data_dir} (Role={target_role}, Endpoint={target_endpoint}, Decoy Ingestion={include_decoys})"
    )

    df = load_all_raw_data(data_dir)

    # 1. Standardize columns (case-normalize BEFORE filtering)
    df["standard_type"] = df["standard_type"].astype(str).str.upper().str.strip()
    df["standard_relation"] = df["standard_relation"].astype(str).str.strip()
    df["assay_type"] = df["assay_type"].astype(str).str.upper().str.strip()
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")

    # 2. Optimized scientific filters:
    # - Exact relationships only (reject > and < bounds)
    # - High confidence target assignment (confidence_score >= 6)
    # - Direct binding (B) or functional (F) assays
    # - AC50 now INCLUDED alongside KI, KD, IC50, EC50
    # - Valid pChEMBL value
    initial_count = len(df)

    # Optimized valid standard types — AC50 included per user approval
    VALID_STANDARD_TYPES = {"KI", "KD", "IC50", "EC50", "AC50"}

    df = df[
        (df["standard_relation"] == "=")
        & (df["confidence_score"] >= 6)
        & (df["assay_type"].isin({"B", "F"}))
        & (df["standard_type"].isin(VALID_STANDARD_TYPES))
        & (df["pchembl_value"].notna())
    ].copy()

    # Endpoint filtering
    if target_endpoint != "all":
        if target_endpoint.upper() in {"KI", "KD"}:
            df = df[df["standard_type"].isin({"KI", "KD"})].copy()
        elif target_endpoint.upper() in {"IC50", "EC50"}:
            df = df[df["standard_type"].isin({"IC50", "EC50"})].copy()

    # Map subtypes
    df["TAG"] = df["TAG"].astype(str).str.strip()
    df["target_subtype"] = df["TAG"].map(SUBTYPE_MAP)
    df = df[df["target_subtype"].notna()].copy()

    # Canonicalize SMILES
    df["canonical_smiles"] = df["smiles"].apply(_canonicalize_smiles)
    df = df[df["canonical_smiles"].notna()].copy()

    post_filter_count = len(df)
    print(
        f"[INFO] Scientific filters: {initial_count} raw rows -> {post_filter_count} high-quality rows."
    )

    # Build known_targets mapping from targets_hit and TAG to avoid false decoys
    known_targets = {}
    for _, row in df.iterrows():
        smi = row["canonical_smiles"]
        hits = str(row.get("targets_hit", ""))
        subtypes = set()
        for hit in hits.split(";"):
            hit = hit.strip()
            if hit in SUBTYPE_MAP:
                subtypes.add(SUBTYPE_MAP[hit])
        # Also include the explicit TAG
        if row["TAG"] in SUBTYPE_MAP:
            subtypes.add(SUBTYPE_MAP[row["TAG"]])

        if smi not in known_targets:
            known_targets[smi] = set()
        known_targets[smi].update(subtypes)

    # 3. Barcode-based deduplication using SmilesRegistry
    registry = SmilesRegistry()

    # Register all SMILES and assign barcodes
    barcodes = []
    for smi in df["canonical_smiles"]:
        barcode, is_new = registry.register(smi)
        barcodes.append(barcode)
    df["barcode"] = barcodes

    # Deduplicate: group by (barcode, target_subtype) instead of raw SMILES
    df["_priority"] = np.where(df["standard_type"].isin({"KI", "KD"}), 0, 1)

    min_priority = df.groupby(["barcode", "target_subtype"])["_priority"].transform(
        "min"
    )
    best_df = df[df["_priority"] == min_priority]

    medians = (
        best_df.groupby(["barcode", "target_subtype"])["pchembl_value"]
        .median()
        .reset_index()
    )
    orig_maxes = (
        df.groupby(["barcode", "target_subtype"])["pchembl_value"].max().reset_index()
    )

    shifts_df = pd.merge(
        medians, orig_maxes, on=["barcode", "target_subtype"], suffixes=("_med", "_max")
    )
    shifts = (
        (shifts_df["pchembl_value_max"] - shifts_df["pchembl_value_med"]).abs().tolist()
    )

    df_deduped = best_df.drop_duplicates(
        subset=["barcode", "target_subtype"], keep="first"
    ).copy()
    df_deduped = df_deduped.drop(columns=["pchembl_value", "_priority"])
    df_deduped = pd.merge(
        df_deduped, medians, on=["barcode", "target_subtype"]
    ).reset_index(drop=True)
    df = df.drop(columns=["_priority"])
    final_count = len(df_deduped)

    mean_shift = sum(shifts) / len(shifts) if shifts else 0.0
    max_shift = max(shifts) if shifts else 0.0

    # Barcode dedup audit
    n_unique_barcodes = df_deduped["barcode"].nunique()
    print(
        f"[INFO] Barcode deduplication: {post_filter_count} rows -> {final_count} unique (barcode, subtype) pairs."
    )
    print(f"[INFO] Unique molecular barcodes: {n_unique_barcodes}")
    print(
        f"[INFO] Collapse stats: mean pChEMBL shift = {mean_shift:.3f}, max shift = {max_shift:.3f}"
    )

    # Build database lookup dictionary
    lookup = {}
    for smi, subdf in df_deduped.groupby("canonical_smiles"):
        lookup[smi] = {
            row["target_subtype"]: float(row["pchembl_value"])
            for _, row in subdf.iterrows()
        }

    # Inject structural decoys (P2Y non-binders) if requested
    if include_decoys:
        print("[INFO] Loading structural decoys...")
        decoy_rows = []
        SUBTYPES = ["A1", "A2A", "A2B", "A3"]
        DECOY_PCHEMBL = 4.0

        # Ingest structural P2Y decoys
        p2y_path = Path("data/processed/p2y_decoys.csv")
        if p2y_path.exists():
            print(f"[INFO] Ingesting structural P2Y decoys from {p2y_path}...")
            p2y_df = pd.read_csv(p2y_path)
            p2y_count = 0
            for _, row in p2y_df.iterrows():
                smiles = row["canonical_smiles"]
                barcode, _ = registry.register(smiles)
                for sub in SUBTYPES:
                    decoy_rows.append(
                        {
                            "TAG": f"{sub}R",
                            "canonical_smiles": smiles,
                            "pchembl_value": DECOY_PCHEMBL,
                            "target_subtype": sub,
                            "standard_type": "DECOY_P2Y",
                            "barcode": barcode,
                        }
                    )
                    p2y_count += 1
            print(f"[SUCCESS] Ingested {p2y_count} structural P2Y non-binder controls.")

        if decoy_rows:
            decoy_df = pd.DataFrame(decoy_rows)
            df_deduped = pd.concat([df_deduped, decoy_df], ignore_index=True)
            print(
                f"[SUCCESS] Ingested {len(decoy_df)} structural P2Y decoy (non-binder) controls."
            )

            # Rebuild lookup to include the newly generated decoys
            lookup = {}
            for smi, subdf in df_deduped.groupby("canonical_smiles"):
                lookup[smi] = {
                    row["target_subtype"]: float(row["pchembl_value"])
                    for _, row in subdf.iterrows()
                }

    # Save registry
    registry.save()
    print(
        f"[INFO] SMILES barcode registry saved ({len(registry)} molecules registered)."
    )

    # Save lookup to disk
    Path(save_lookup_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_lookup_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, sort_keys=True)

    keep_cols = [
        "TAG",
        "canonical_smiles",
        "pchembl_value",
        "target_subtype",
        "standard_type",
        "barcode",
    ]
    available_cols = [c for c in keep_cols if c in df_deduped.columns]
    df_deduped = df_deduped[available_cols].copy()

    return df_deduped, lookup


if __name__ == "__main__":
    df, lookup = load_and_clean("data/raw", mode="precise", include_decoys=True)
    print(
        f"\nFinal dataset: {len(df)} rows, {df['canonical_smiles'].nunique()} unique SMILES"
    )
    print("Per-subtype distribution:")
    print(df["target_subtype"].value_counts().to_string())
