import json
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
    mode: str = "standard",
):
   
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
    df["standard_type"] = df["standard_type"].astype(str).str.upper().str.strip()

    if mode == "standard":
        df = df[df["standard_type"] != "IC50"].copy()

    elif mode == "strict":
        df["standard_relation"] = df["standard_relation"].astype(str).str.strip()
        df["standard_units"] = df["standard_units"].astype(str).str.strip()
        df["assay_type"] = df["assay_type"].astype(str).str.strip()

        df = df[df["standard_relation"] == "="].copy()
        df = df[df["standard_units"].str.lower() == "nm"].copy()
        df = df[df["assay_type"].str.upper() == "B"].copy()

        df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
        df = df[df["confidence_score"].notna() & (df["confidence_score"] >= 7)].copy()
        df = df[df["standard_type"].isin({"KI", "KD"})].copy()

    else:
        raise ValueError("mode must be 'standard' or 'strict'")

    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df[df["pchembl_value"].notna()].copy()

    df["TAG"] = df["TAG"].astype(str).str.strip()
    df["target_subtype"] = df["TAG"].map(SUBTYPE_MAP)
    df = df[df["target_subtype"].notna()].copy()

    df["canonical_smiles"] = df["smiles"].apply(_canonicalize_smiles)
    df = df[df["canonical_smiles"].notna()].copy()

    df = (
        df.sort_values("pchembl_value", ascending=False)
        .drop_duplicates(subset=["canonical_smiles", "target_subtype"], keep="first")
        .reset_index(drop=True)
    )

    lookup = {}
    for smi, subdf in df.groupby("canonical_smiles"):
        lookup[smi] = {
            row["target_subtype"]: float(row["pchembl_value"])
            for _, row in subdf.iterrows()
        }

    Path(save_lookup_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_lookup_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, sort_keys=True)

    keep_cols = ["TAG", "canonical_smiles", "pchembl_value", "target_subtype", "standard_type"]
    df = df[keep_cols].copy()

    return df, lookup
