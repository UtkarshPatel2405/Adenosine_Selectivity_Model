import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("=== NOVEL EXTERNAL VALIDATION SET ANALYSIS ===\n")

# Load ground truth and predictions
truth_df = pd.read_csv("data/processed/novel_test_set.csv")
pred_df = pd.read_csv("data/processed/novel_test_result.csv")

# Ensure they align by canonical smiles
merged = pd.merge(truth_df, pred_df, on="canonical_smiles", suffixes=('_true', '_pred'))

subtypes = ["A1", "A2A", "A2B", "A3"]

print("1. ABSOLUTE ACTIVITY EVALUATION (RMSE & MAE)")
print("-" * 50)
for st in subtypes:
    true_col = f"{st}_true"
    pred_col = f"{st}_pred"
    
    # Filter out NaNs for this specific subtype in ground truth
    mask = merged[true_col].notna() & merged[pred_col].notna()
    y_true = merged.loc[mask, true_col]
    y_pred = merged.loc[mask, pred_col]
    
    if len(y_true) > 0:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        fold_error = np.mean(10 ** np.abs(y_true - y_pred))
        print(f"Subtype {st:3s} (n={len(y_true):3d}): RMSE = {rmse:.3f}, MAE = {mae:.3f} log units (Avg Fold Error = {fold_error:.1f}x)")
    else:
        print(f"Subtype {st:3s} (n=  0): No data")

print("\n2. SELECTIVITY PREDICTION EVALUATION")
print("-" * 50)
# Selectivity: Does the model correctly identify the receptor with the highest pChEMBL value?
# Note: Many molecules only have experimental data for 1 or 2 receptors.
# True selectivity is only meaningful if they were tested against multiple.

# Let's count how many have data for >= 2 receptors
counts = truth_df[subtypes].notna().sum(axis=1)
multi_target_idx = counts[counts >= 2].index
multi_target = merged.loc[merged.index.isin(multi_target_idx)].copy()

print(f"Molecules experimentally tested on multiple receptors: {len(multi_target)}")

if len(multi_target) > 0:
    correct_selectivity_top1 = 0
    correct_selectivity_top2 = 0
    valid_top2_comparisons = 0
    for idx, row in multi_target.iterrows():
        # Get true values, ignoring NaNs
        true_vals = {st: row[f"{st}_true"] for st in subtypes if pd.notna(row[f"{st}_true"])}
        if not true_vals: continue
        
        # The true preferred target is the one with the highest value
        true_best = max(true_vals, key=true_vals.get)
        
        # The predicted preferred target (only comparing among those experimentally tested!)
        pred_vals = {st: row[f"{st}_pred"] for st in true_vals.keys()}
        
        # Sort predicted targets from best to worst
        pred_sorted = sorted(pred_vals.keys(), key=lambda k: pred_vals[k], reverse=True)
        pred_best = pred_sorted[0]
        
        if true_best == pred_best:
            correct_selectivity_top1 += 1
            
        # Top-2 only makes sense if at least 3 targets were tested experimentally
        if len(pred_sorted) >= 3: 
            valid_top2_comparisons += 1
            if true_best in pred_sorted[:2]:
                correct_selectivity_top2 += 1
            
    acc_top1 = correct_selectivity_top1 / len(multi_target) * 100
    print(f"Recall@1 (Top-1 Accuracy): {correct_selectivity_top1}/{len(multi_target)} ({acc_top1:.1f}%)")
    
    if valid_top2_comparisons > 0:
        acc_top2 = correct_selectivity_top2 / valid_top2_comparisons * 100
        print(f"Recall@2 (Among molecules with >=3 targets tested): {correct_selectivity_top2}/{valid_top2_comparisons} ({acc_top2:.1f}%)")

print("\n3. RELIABILITY (APPLICABILITY DOMAIN)")
print("-" * 50)
high_rel = (merged['reliability'] >= 0.6).sum()
med_rel = ((merged['reliability'] >= 0.4) & (merged['reliability'] < 0.6)).sum()
low_rel = (merged['reliability'] < 0.4).sum()
print(f"Average Tanimoto similarity to training data: {merged['reliability'].mean():.3f}")
print(f"High Reliability (>=0.6): {high_rel} molecules")
print(f"Med  Reliability (0.4-0.6): {med_rel} molecules")
print(f"Low  Reliability (<0.4) : {low_rel} molecules")
