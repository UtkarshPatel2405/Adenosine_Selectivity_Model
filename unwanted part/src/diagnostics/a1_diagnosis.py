import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.data_loader import load_and_clean
from src.scaffold_split import _murcko_scaffold_smiles

sns.set_style("whitegrid")

def run_subtype_diagnosis(subtype: str, df_sub: pd.DataFrame, out_dir: Path):
    subtype_lower = subtype.lower()
    n_total = len(df_sub)
    print(f"[INFO] Analyzing {subtype} subtype with {n_total} compounds...")
    if n_total == 0:
        return None
        
    pchembl_vals = df_sub["pchembl_value"].values
    
    # 1. Plot pChEMBL distribution
    plt.figure(figsize=(8, 5))
    color_map = {"A1": "teal", "A2A": "royalblue", "A2B": "forestgreen", "A3": "purple"}
    color = color_map.get(subtype, "teal")
    
    sns.histplot(pchembl_vals, kde=True, bins=25, color=color, edgecolor="black", alpha=0.7)
    plt.axvline(np.mean(pchembl_vals), color="crimson", linestyle="--", lw=2, label=f"Mean ({np.mean(pchembl_vals):.2f})")
    plt.axvline(np.median(pchembl_vals), color="darkorange", linestyle="-.", lw=2, label=f"Median ({np.median(pchembl_vals):.2f})")
    plt.title(f"{subtype} Receptor pChEMBL Distribution Analysis", fontsize=13, fontweight="bold")
    plt.xlabel("pChEMBL Value", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.legend()
    plt.tight_layout()
    dist_plot_file = out_dir / f"{subtype_lower}_pchembl_distribution.png"
    plt.savefig(dist_plot_file, dpi=300)
    plt.close()
    
    # 2. Scaffold analysis
    scaffolds = df_sub["canonical_smiles"].apply(_murcko_scaffold_smiles).values
    df_sub = df_sub.copy()
    df_sub["_scaffold"] = scaffolds
    unique_scaffolds = set(scaffolds) - {"__INVALID__", ""}
    n_scaffolds = len(unique_scaffolds)
    diversity_ratio = n_scaffolds / n_total if n_total > 0 else 0.0
    
    scaf_counts = df_sub["_scaffold"].value_counts()
    top_scaf_sizes = scaf_counts.head(5).to_dict()
    
    # 3. Standard type breakdown
    type_breakdown = df_sub["standard_type"].value_counts().to_dict()
    
    # 4. Detect Activity Cliffs using fast rdkit BulkTanimotoSimilarity
    mols = [Chem.MolFromSmiles(s) for s in df_sub["canonical_smiles"].values]
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(m) for m in mols]
    
    activity_cliffs = []
    for i in range(n_total):
        if i == n_total - 1:
            break
        qfp = fps[i]
        tfps = fps[i+1:]
        sims = DataStructs.BulkTanimotoSimilarity(qfp, tfps)
        for idx, sim in enumerate(sims):
            j = i + 1 + idx
            if sim >= 0.80:
                diff = abs(pchembl_vals[i] - pchembl_vals[j])
                if diff >= 1.50:
                    activity_cliffs.append({
                        "compound_1_smiles": df_sub.loc[i, "canonical_smiles"],
                        "compound_1_pchembl": float(pchembl_vals[i]),
                        "compound_2_smiles": df_sub.loc[j, "canonical_smiles"],
                        "compound_2_pchembl": float(pchembl_vals[j]),
                        "tanimoto_similarity": float(sim),
                        "pchembl_difference": float(diff)
                    })
                    
    n_cliffs = len(activity_cliffs)
    
    report_data = {
        "subtype": subtype,
        "n_compounds": n_total,
        "pchembl_stats": {
            "mean": float(np.mean(pchembl_vals)),
            "std": float(np.std(pchembl_vals)),
            "min": float(np.min(pchembl_vals)),
            "max": float(np.max(pchembl_vals)),
            "median": float(np.median(pchembl_vals))
        },
        "scaffold_diversity": {
            "n_unique_scaffolds": n_scaffolds,
            "diversity_ratio": diversity_ratio,
            "top_scaffold_sizes": top_scaf_sizes
        },
        "standard_type_breakdown": {str(k): int(v) for k, v in type_breakdown.items()},
        "n_activity_cliffs": n_cliffs,
        "activity_cliffs": activity_cliffs[:20]
    }
    
    # Save report
    report_file = out_dir / f"{subtype_lower}_diagnosis_report.json"
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)
        
    # Plot cliffs distribution
    if n_cliffs > 0:
        plt.figure(figsize=(7, 4))
        cliff_diffs = [c["pchembl_difference"] for c in activity_cliffs]
        sns.histplot(cliff_diffs, color="crimson", kde=True, bins=10, edgecolor="black", alpha=0.7)
        plt.title(f"Distribution of pChEMBL Shifts in {subtype} Activity Cliffs", fontsize=12, fontweight="bold")
        plt.xlabel("|ΔpChEMBL| Shift", fontsize=11)
        plt.ylabel("Count", fontsize=11)
        plt.tight_layout()
        cliff_plot_file = out_dir / f"{subtype_lower}_activity_cliffs_shifts.png"
        plt.savefig(cliff_plot_file, dpi=300)
        plt.close()
        
    return report_data

def run_combined_diagnosis(df: pd.DataFrame, out_dir: Path):
    n_total = len(df)
    print(f"[INFO] Analyzing combined receptor dataset with {n_total} compounds...")
    pchembl_vals = df["pchembl_value"].values
    
    # 1. Combined pChEMBL distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(pchembl_vals, kde=True, bins=25, color="dimgrey", edgecolor="black", alpha=0.7)
    plt.axvline(np.mean(pchembl_vals), color="crimson", linestyle="--", lw=2, label=f"Mean ({np.mean(pchembl_vals):.2f})")
    plt.axvline(np.median(pchembl_vals), color="darkorange", linestyle="-.", lw=2, label=f"Median ({np.median(pchembl_vals):.2f})")
    plt.title("Combined Adenosine Receptor pChEMBL Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("pChEMBL Value", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.legend()
    plt.tight_layout()
    dist_plot_file = out_dir / "combined_pchembl_distribution.png"
    plt.savefig(dist_plot_file, dpi=300)
    plt.close()
    
    # 2. Scaffold analysis
    scaffolds = df["canonical_smiles"].apply(_murcko_scaffold_smiles).values
    df = df.copy()
    df["_scaffold"] = scaffolds
    unique_scaffolds = set(scaffolds) - {"__INVALID__", ""}
    n_scaffolds = len(unique_scaffolds)
    diversity_ratio = n_scaffolds / n_total if n_total > 0 else 0.0
    
    scaf_counts = df["_scaffold"].value_counts()
    top_scaf_sizes = scaf_counts.head(5).to_dict()
    
    # 3. Standard Type breakdown
    type_breakdown = df["standard_type"].value_counts().to_dict()
    
    # 4. Target Subtype breakdown
    subtype_breakdown = df["target_subtype"].value_counts().to_dict()
    
    report_data = {
        "n_compounds": n_total,
        "pchembl_stats": {
            "mean": float(np.mean(pchembl_vals)),
            "std": float(np.std(pchembl_vals)),
            "min": float(np.min(pchembl_vals)),
            "max": float(np.max(pchembl_vals)),
            "median": float(np.median(pchembl_vals))
        },
        "scaffold_diversity": {
            "n_unique_scaffolds": n_scaffolds,
            "diversity_ratio": diversity_ratio,
            "top_scaffold_sizes": top_scaf_sizes
        },
        "standard_type_breakdown": {str(k): int(v) for k, v in type_breakdown.items()},
        "target_subtype_breakdown": {str(k): int(v) for k, v in subtype_breakdown.items()},
    }
    
    report_file = out_dir / "combined_diagnosis_report.json"
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)
        
    return report_data

def run_a1_diagnosis(data_path: str = "data/raw/AR_all_unique_parents_with_smiles.csv"):
    print("\n" + "="*60)
    print("RUNNING ADENOSINE RECEPTOR DATA QUALITY DIAGNOSTICS")
    print("="*60)
    
    # Load clean precise data with a custom lookup path to avoid overwriting master db_lookup.json
    df, _ = load_and_clean(data_path, mode="precise", save_lookup_path="data/processed/db_lookup_actives_only.json")
    
    out_dir = Path("outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Run individual subtype diagnoses
    subtypes = ["A1", "A2A", "A2B", "A3"]
    for subtype in subtypes:
        df_sub = df[df["target_subtype"] == subtype].copy().reset_index(drop=True)
        run_subtype_diagnosis(subtype, df_sub, out_dir)
        
    # 2. Run combined diagnosis
    run_combined_diagnosis(df, out_dir)
    print("\n" + "="*60)
    print("[SUCCESS] COMPLETED DIAGNOSTICS FOR ALL SUBTYPES & COMBINED")
    print("="*60)

if __name__ == "__main__":
    run_a1_diagnosis()
