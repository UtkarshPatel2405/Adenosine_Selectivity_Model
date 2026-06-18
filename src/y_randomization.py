"""
Y-Randomization Validation — Label shuffling test to prove genuine chemical SAR.

Now supports ALL 4 subtypes (A1, A2A, A2B, A3) via --all flag.
"""

import json
from pathlib import Path
import numpy as np
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

from src.data_loader import load_and_clean
from src.features import build_feature_matrix

sns.set_style("whitegrid")

SUBTYPES = ["A1", "A2A", "A2B", "A3"]


def run_y_randomization(
    subtype: str, data_path: str = "data/raw", n_iterations: int = 20
):
    print("\n" + "=" * 60)
    print(f"Y-RANDOMIZATION TEST FOR {subtype} (n={n_iterations})")
    print("=" * 60)

    # 1. Load Data
    df, _ = load_and_clean(
        data_path,
        mode="precise",
        save_lookup_path="data/processed/db_lookup_actives_only.json",
        include_decoys=False,
    )
    df_st = df[df["target_subtype"] == subtype].copy().reset_index(drop=True)

    if len(df_st) < 50:
        print(f"[ERROR] Insufficient data for {subtype} ({len(df_st)} samples)")
        return None

    # 2. Use global scaffold split from production training
    split_path = Path("data/processed/global_split.json")
    with open(split_path) as f:
        split = json.load(f)
    train_smiles = set(split["train"])
    test_smiles = set(split["test"])

    # Filter to SMILES present in our data (handles include_decoys mismatch)
    available_smiles = set(df_st["canonical_smiles"])
    train_smiles = train_smiles & available_smiles
    test_smiles = test_smiles & available_smiles

    df_st_split = df_st[df_st["canonical_smiles"].isin(train_smiles | test_smiles)]
    train_df = df_st_split[df_st_split["canonical_smiles"].isin(train_smiles)].copy()
    test_df = df_st_split[df_st_split["canonical_smiles"].isin(test_smiles)].copy()

    train_df = train_df.rename(columns={"canonical_smiles": "smiles"})
    test_df = test_df.rename(columns={"canonical_smiles": "smiles"})

    # Build standard features
    X_train, X_test, _ = build_feature_matrix(
        train_df, test_df, smiles_col="smiles", save_to_disk=False
    )
    y_train = train_df["pchembl_value"].values
    y_test = test_df["pchembl_value"].values

    # Train standard model to get the "Real R²"
    base_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    base_model.fit(X_train, y_train)
    real_preds = base_model.predict(X_test)
    real_r2 = float(r2_score(y_test, real_preds))
    print(f"[INFO] Real Model R² Score: {real_r2:.4f}")

    # 3. Y-Randomization loop
    shuffled_r2s = []

    for i in range(n_iterations):
        # Shuffle the y values to break the SMILES-activity relationship
        y_train_shuffled = np.random.RandomState(42 + i).permutation(y_train)

        # Train model on shuffled training set
        shuffled_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=42 + i,
        )
        shuffled_model.fit(X_train, y_train_shuffled)

        # Predict on unshuffled test set (to see if model can learn noise)
        shuffled_preds = shuffled_model.predict(X_test)
        shuffled_r2 = float(r2_score(y_test, shuffled_preds))
        shuffled_r2s.append(shuffled_r2)

        if (i + 1) % 5 == 0:
            print(
                f"  Completed iteration {i + 1}/{n_iterations} | R² = {shuffled_r2:.4f}"
            )

    mean_shuffled_r2 = float(np.mean(shuffled_r2s))
    std_shuffled_r2 = float(np.std(shuffled_r2s))

    print(f"\n[RESULTS] Y-Randomization Summary for {subtype}:")
    print(f"  Real R²:              {real_r2:.4f}")
    print(f"  Shuffled R²:          {mean_shuffled_r2:.4f} ± {std_shuffled_r2:.4f}")

    # Check for target leakage or model learning spurious patterns
    leakage_flag = False
    if mean_shuffled_r2 > 0.10:
        print(
            "[WARNING] Shuffled R² > 0.10! Possible target leakage or severe overfitting to background features."
        )
        leakage_flag = True
    else:
        print(
            "[SUCCESS] Shuffled R² is near-zero/negative. The model relies on actual chemical structures rather than noise."
        )

    # Save outputs
    out_dir = Path("outputs/y_randomization")
    out_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "subtype": subtype,
        "n_iterations": n_iterations,
        "real_r2": real_r2,
        "shuffled_r2_mean": mean_shuffled_r2,
        "shuffled_r2_std": std_shuffled_r2,
        "shuffled_r2_values": shuffled_r2s,
        "leakage_warning": leakage_flag,
    }

    with open(out_dir / f"{subtype}_report.json", "w") as f:
        json.dump(report_data, f, indent=2)

    # Plot distribution
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        shuffled_r2s, fill=True, label="Shuffled Labels R²", color="skyblue", lw=2
    )
    plt.axvline(
        real_r2,
        color="crimson",
        linestyle="--",
        lw=2.5,
        label=f"Real Model R² ({real_r2:.3f})",
    )
    plt.axvline(0.0, color="gray", linestyle="-", alpha=0.5)
    plt.title(
        f"Y-Randomization Test: {subtype} subtype", fontsize=13, fontweight="bold"
    )
    plt.xlabel("R² Score", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plot_file = out_dir / f"{subtype}_distribution.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved Y-Randomization distribution plot to {plot_file}")

    return report_data


def run_all_subtypes(n_iterations: int = 20):
    """Run Y-randomization for ALL 4 receptor subtypes."""
    print("\n" + "=" * 60)
    print("Y-RANDOMIZATION VALIDATION FOR ALL SUBTYPES")
    print("=" * 60)

    all_results = {}
    for st in SUBTYPES:
        result = run_y_randomization(subtype=st, n_iterations=n_iterations)
        if result is not None:
            all_results[st] = result

    # Save combined summary
    out_dir = Path("outputs/y_randomization")
    summary = {
        "n_subtypes_validated": len(all_results),
        "subtypes": list(all_results.keys()),
        "summary": {
            st: {
                "real_r2": r["real_r2"],
                "shuffled_r2_mean": r["shuffled_r2_mean"],
                "shuffled_r2_std": r["shuffled_r2_std"],
                "leakage_warning": r["leakage_warning"],
            }
            for st, r in all_results.items()
        },
    }
    with open(out_dir / "all_subtypes_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\n[SUCCESS] All-subtype Y-randomization summary saved to {out_dir / 'all_subtypes_summary.json'}"
    )

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Y-Randomization validation test")
    parser.add_argument(
        "--subtype", default=None, help="Subtype to validate (or use --all)"
    )
    parser.add_argument(
        "--iterations", type=int, default=20, help="Number of shuffling runs"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run Y-randomization for ALL 4 subtypes"
    )
    args = parser.parse_args()

    if args.all:
        run_all_subtypes(n_iterations=args.iterations)
    elif args.subtype:
        run_y_randomization(subtype=args.subtype, n_iterations=args.iterations)
    else:
        # Default: run all subtypes
        run_all_subtypes(n_iterations=args.iterations)
