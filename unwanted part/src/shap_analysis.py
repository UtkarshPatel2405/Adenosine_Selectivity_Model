import os
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.predictor import SUBTYPES

sns.set_style("whitegrid")

def run_shap_analysis(subtype: str = "A2A", mode: str = "precise"):
    print(f"\n" + "="*60)
    print(f"SHAP TREE ANALYSIS FOR {subtype} ({mode} mode)")
    print("="*60)
    
    # 1. Load trained model
    model_path = Path(f"models/{mode}/xgboost_{mode}_{subtype.lower()}_model.pkl")
    if not model_path.exists():
        model_path = Path(f"models/xgboost_{subtype.lower()}_model.pkl")
    if not model_path.exists():
        print(f"[ERROR] No model found for {subtype} at {model_path}. Complete production retraining first.")
        return
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Extract fitted base estimator for SHAP TreeExplainer
    if type(model).__name__ == "CrossConformalRegressor":
        estimator = model._mapie_regressor.estimator_.estimators_[0]
    elif type(model).__name__ == "MapieRegressor":
        estimator = model.estimators_[0]
    elif isinstance(model, list) and len(model) > 0:
        estimator = model[0]
    else:
        estimator = model
        
    # 2. Load test features and feature names
    features_test_path = Path("data/processed/features_test.pkl")
    scaler_path = Path("models/scaler.pkl")
    
    if not features_test_path.exists() or not scaler_path.exists():
        print("[ERROR] Features or Scaler missing. Feature matrix generation must be run first.")
        return
        
    with open(features_test_path, "rb") as f:
        X_test = pickle.load(f)
        
    with open(scaler_path, "rb") as f:
        pipeline = pickle.load(f)
        
    # Build complete feature names list
    feature_names = [f"Morgan_FP_{i}" for i in range(2048)] + [f"MACCS_{i}" for i in range(167)]
    selected_desc_names = pipeline.feature_filter.feature_names
    feature_names.extend(selected_desc_names)
    
    print(f"[INFO] Loaded test set with shape: {X_test.shape}")
    print(f"  Total features: {len(feature_names)}")
    
    # 3. Create TreeExplainer
    print("[INFO] Initializing SHAP TreeExplainer and calculating SHAP values...")
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer(X_test)
    
    out_dir = Path("outputs/shap")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f"SHAP Global Feature Importance (Beeswarm): {subtype}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    beeswarm_file = out_dir / f"{subtype}_beeswarm.png"
    plt.savefig(beeswarm_file, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved summary beeswarm plot to {beeswarm_file}")
    
    # Generate Bar plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title(f"SHAP Global Feature Importance (Bar): {subtype}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    bar_file = out_dir / f"{subtype}_bar.png"
    plt.savefig(bar_file, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved summary bar plot to {bar_file}")
    
    # Analyze the most important features to perform the Chemical Sanity Check
    mean_abs_shaps = np.abs(shap_values.values).mean(axis=0)
    sorted_indices = np.argsort(mean_abs_shaps)[::-1]
    
    top_features = []
    print(f"\n[INFO] Chemical Sanity Check - Top 10 Most Important Features:")
    for rank, idx in enumerate(sorted_indices[:10], 1):
        name = feature_names[idx]
        val = mean_abs_shaps[idx]
        top_features.append({"rank": rank, "feature": name, "mean_abs_shap": float(val)})
        print(f"  {rank}. {name}: {val:.4f}")
        
    # Standard chemical analysis for Adenosine Receptor selectivities:
    # Aromatic rings (pi-pi stacking in the binding pocket), hydrogen bond patterns (AromRings, Lipinski, HBD, HBA),
    # and specific steric properties (MW, TPSA) are expected top features.
    expected_top = ["AromRings", "HBD", "HBA", "TPSA", "LogP", "MolWt"]
    matching_expected = [f for f in expected_top if any(f in item["feature"] for item in top_features)]
    
    sanity_status = "PASS"
    sanity_message = "Top features are dominated by structurally relevant physicochemical properties."
    if len(matching_expected) == 0:
        sanity_status = "WARNING"
        sanity_message = "Top features are dominated by isolated fingerprint bits rather than global physicochemical descriptors. Verify model is not learning noise."
        
    print(f"\n[STATUS] Chemical Sanity: {sanity_status} - {sanity_message}")
    
    report_data = {
        "subtype": subtype,
        "mode": mode,
        "top_features": top_features,
        "sanity_check": {
            "status": sanity_status,
            "message": sanity_message,
            "expected_features_found": matching_expected
        }
    }
    
    with open(out_dir / f"{subtype}_shap_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"[SUCCESS] Saved SHAP report to {out_dir}/{subtype}_shap_report.json")
    
    # 4. Generate dependence plots for top 3 continuous descriptors if present in top features
    continuous_desc_top = [feature_names[i] for i in sorted_indices if not (feature_names[i].startswith("Morgan_FP_") or feature_names[i].startswith("MACCS_"))][:3]
    
    for desc_name in continuous_desc_top:
        desc_idx = feature_names.index(desc_name)
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(desc_idx, shap_values.values, X_test, feature_names=feature_names, show=False)
        plt.title(f"SHAP Dependence: {desc_name} on {subtype}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        dep_file = out_dir / f"{subtype}_dependence_{desc_name}.png"
        plt.savefig(dep_file, dpi=300)
        plt.close()
        print(f"[SUCCESS] Saved dependence plot for {desc_name} to {dep_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SHAP Tree Explainer validation")
    parser.add_argument("--subtype", default="A2A", help="Subtype to explain")
    parser.add_argument("--mode", default="precise", help="Model mode")
    args = parser.parse_args()
    
    run_shap_analysis(subtype=args.subtype, mode=args.mode)
