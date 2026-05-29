from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import shap
import pickle
import matplotlib.pyplot as plt

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predictor import SUBTYPES, predict
from src.chem_utils import topk_tanimoto
from src.app.components.structure_viz import draw_2d, draw_2d_svg, generate_3d_conformer
from src.app.components.pains_checker import check_pains
from src.app.components.drug_likeness import qed_profile
from src.app.components.applicability_domain import nearest_tanimoto
from src.app.components.batch_predict import predict_batch, _infer_smiles_col
from src.app.components.model_reports import (
    load_evaluation_tables,
    load_fingerprint_comparison,
    load_scaffold_ood,
    load_run_summary,
    load_mode_examples,
    load_examples,
    outputs_exist,
)

def explain_feature_chemically(name: str, val: float, smiles: str) -> dict:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    
    desc_explanations = {
        "MolLogP": "Lipophilicity (Octanol-water partition coefficient). A higher value increases hydrophobic interactions but may reduce solubility.",
        "TPSA": "Topological Polar Surface Area. Represents molecular polarity; crucial for cell membrane permeability and BBB penetration.",
        "MolWt": "Molecular Weight. Represents molecular size. High weight can decrease oral absorption (Lipinski's Rule of 5).",
        "NumHDonors": "Hydrogen Bond Donors. The number of NH or OH groups in the molecule.",
        "NumHAcceptors": "Hydrogen Bond Acceptors. The number of nitrogen or oxygen atoms with lone pairs.",
        "NumRotatableBonds": "Rotatable Bonds. Indicates molecular flexibility. Lower flexibility often reduces binding entropy loss.",
        "NumAromaticRings": "Aromatic Rings. Direct participant in pi-pi stacking interactions with target receptor binding pockets.",
        "FractionCSP3": "Saturated Carbons (sp3 fraction). Indicates molecular 3D complexity and saturation; correlated with better solubility.",
        "MolMR": "Molecular Refractivity. Represents molecular volume and polarizability.",
        "MaxAbsPartialCharge": "Maximum absolute partial charge. High charge density can affect electrostatic binding interactions.",
        "MinPartialCharge": "Minimum partial charge. Represents the most negatively charged region, often active in hydrogen bonding."
    }
    
    if name in desc_explanations:
        return {
            "Type": "Physicochemical Property",
            "Property": name,
            "Query Value": f"{val:.3f}" if isinstance(val, (int, float)) else str(val),
            "Interpretation": desc_explanations[name]
        }
    
    if not (name.startswith("Morgan_FP_") or name.startswith("MACCS_")):
        clean_name = name.replace("_", " ")
        return {
            "Type": "Physicochemical Property",
            "Property": name,
            "Query Value": f"{val:.3f}" if isinstance(val, (int, float)) else str(val),
            "Interpretation": f"Continuous descriptor '{clean_name}' calculated by RDKit representing molecular structure topology or charge."
        }
        
    maccs_dict = {
        "MACCS_115": "Presence of CH3 (methyl) or terminal alkyl group.",
        "MACCS_137": "Presence of a Carbonyl group (C=O).",
        "MACCS_139": "Presence of a primary or secondary amine.",
        "MACCS_143": "Presence of a N-C-O or similar polar linker fragment.",
        "MACCS_153": "Presence of a C=C double bond (alkene).",
        "MACCS_155": "Presence of Halogen atoms (F, Cl, Br, I).",
        "MACCS_160": "Presence of a CH3 group.",
        "MACCS_164": "Presence of one or more Oxygen atoms.",
        "MACCS_165": "Presence of a Ring structure."
    }
    
    if name.startswith("MACCS_"):
        key_num = name.split("_")[1]
        interpretation = maccs_dict.get(name, f"MACCS Key #{key_num} (standard structural fragment pattern).")
        return {
            "Type": "MACCS Structural Key",
            "Property": name,
            "Query Value": "Present (1)" if float(val) > 0.5 else "Absent (0)",
            "Interpretation": interpretation
        }
        
    if name.startswith("Morgan_FP_"):
        bit_idx = int(name.split("_")[2])
        if mol is None:
            return {
                "Type": "Morgan Fingerprint Bit",
                "Property": name,
                "Query Value": "Present (1)" if float(val) > 0.5 else "Absent (0)",
                "Interpretation": f"Morgan circular fingerprint bit #{bit_idx}."
            }
            
        info = {}
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=info)
            if bit_idx in info and len(info[bit_idx]) > 0:
                atom_idx, radius = info[bit_idx][0]
                if radius == 0:
                    symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                    smarts = f"[{symbol}]"
                    desc = f"Single atom: {symbol} environment"
                else:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                    submol = Chem.PathToSubmol(mol, env)
                    smarts = Chem.MolToSmarts(submol)
                    desc = f"Circular environment around atom {mol.GetAtomWithIdx(atom_idx).GetSymbol()} (radius={radius})"
                
                return {
                    "Type": f"Morgan FP Environment (radius={radius})",
                    "Property": name,
                    "Query Value": "Present (1)",
                    "Interpretation": f"{desc}. Exact structural SMARTS matched in this molecule: `{smarts}`"
                }
        except Exception:
            pass
            
        return {
            "Type": "Morgan Fingerprint Bit",
            "Property": name,
            "Query Value": "Present (1)" if float(val) > 0.5 else "Absent (0)",
            "Interpretation": f"Morgan circular fingerprint bit #{bit_idx}."
        }
            
    return {}

def _ad_label(sim: float | None) -> str:
    if sim is None:
        return "Unknown"
    if sim >= 0.6:
        return "High"
    if sim >= 0.4:
        return "Medium"
    return "Low"

def render_3d_viewer(mol_block: str) -> str:
    import json
    escaped_mol = json.dumps(mol_block)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://3dmol.org/build/3Dmol-min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: transparent;
            }}
            #container-3dmol {{
                width: 100%;
                height: 350px;
                position: relative;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                background-color: #f8f9fa;
            }}
        </style>
    </head>
    <body>
        <div id="container-3dmol"></div>
        <script>
            $(document).ready(function() {{
                let element = $('#container-3dmol');
                let config = {{ backgroundColor: '#f8f9fa' }};
                let viewer = $3Dmol.createViewer(element, config);
                let molData = {escaped_mol};
                viewer.addModel(molData, "sdf");
                viewer.setStyle({{}}, {{stick: {{radius: 0.2, colorscheme: 'Jmol'}}, sphere: {{radius: 0.4, scale: 0.3}}}});
                viewer.zoomTo();
                viewer.render();
            }});
        </script>
    </body>
    </html>
    """
    return html_content

def _section_single_prediction():
    st.header("Single SMILES Prediction")

    # User Input - Single Smiles Field
    smiles = st.text_input("SMILES Compound Input", value="CCn1c(/N=C/c2ccc(Br)cc2)c(C#N)sc1=S")
    
    # Model mode is set to unified precise model
    mode = "precise"
    
    threshold = 6.0

    if st.button("Predict"):
        with st.spinner("Generating 3D Conformer & Running Predictions..."):
            # 1. Chemical Structure Visualization (Side-by-side Columns)
            mol_block, min_charge, max_charge = generate_3d_conformer(smiles)
            svg = draw_2d_svg(smiles)
            
            col_2d, col_3d = st.columns(2)
            with col_2d:
                if svg is not None:
                    st.image(svg, caption="2D Vector Depiction", use_container_width=True)
                    st.download_button(
                        label="Download 2D SVG",
                        data=svg,
                        file_name="structure_2d.svg",
                        mime="image/svg+xml"
                    )
                else:
                    st.warning("Could not render 2D structure – is the SMILES valid?")
            with col_3d:
                if mol_block is not None:
                    st.components.v1.html(render_3d_viewer(mol_block), height=360)
                    
                    from src.app.components.structure_viz import generate_pdb_block
                    pdb_block = generate_pdb_block(smiles)
                    
                    c_sdf, c_pdb = st.columns(2)
                    with c_sdf:
                        st.download_button(
                            label="Download 3D SDF",
                            data=mol_block,
                            file_name="conformer_3d.sdf",
                            mime="chemical/x-mdl-sdfile"
                        )
                    with c_pdb:
                        if pdb_block is not None:
                            st.download_button(
                                label="Download 3D PDB",
                                data=pdb_block,
                                file_name="conformer_3d.pdb",
                                mime="chemical/x-pdb"
                            )
                else:
                    st.warning("Could not generate 3D conformer.")

            # 2. Run Prediction Pipeline
            try:
                r = predict(smiles, threshold=threshold, mode=mode)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

            # Define canon variable to avoid NameError in SHAP
            canon = r["smiles"]

            # 3. Data Source Information
            if r["in_database"]:
                st.success("Experimental data retrieved from ChEMBL (Database Hit).")
                st.caption("Note: Missing experimental values for specific subtypes are assumed as 0.000 per experimental protocol.")
            else:
                st.info("ML Ensemble model prediction (Novel Molecule).")
           
            # 4. Physicochemical Profile & Gasteiger Charges
            st.subheader("Physicochemical Profile")
            d = r["descriptors"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mol. Weight", d["MW"])
            c2.metric("LogP", d["LogP"])
            c3.metric("H-Bond Donors", d["HBD"])
            c4.metric("H-Bond Acceptors", d["HBA"])
            
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Rotatable Bonds", d["RotBonds"])
            c6.metric("Aromatic Rings", d["AromRings"])
            c7.metric("TPSA", d["TPSA"])
            
            # Fetch QED score dynamically
            profile = qed_profile(smiles)
            qed_val = profile.get("QED", 0.0) if profile else 0.0
            c8.metric("QED Score", f"{qed_val:.3f}")
            
            c9, c10, _, _ = st.columns(4)
            c9.metric("Min Gasteiger Charge", f"{min_charge:.3f}")
            c10.metric("Max Gasteiger Charge", f"{max_charge:.3f}")

            # 5. Subtype Bioactivity Profile
            st.subheader("Subtype Bioactivity Profile")
            st.write(f"**Primary Target Receptor:** {r['best_target']}")
            preds, unc = r["predictions"], r["uncertainty"]
            
            rows = []
            for k in SUBTYPES:
                p_val = round(float(preds[k]), 3)
                unc_val = round(float(unc[k]), 3)
                if "intervals" in r and k in r["intervals"]:
                    low = round(r["intervals"][k]["lower"], 3)
                    high = round(r["intervals"][k]["upper"], 3)
                    width = round(r["intervals"][k]["width"], 3)
                    interval_str = f"[{low}, {high}]"
                else:
                    interval_str = "N/A"
                    width = "N/A"
                    
                rows.append({
                    "Subtype": k,
                    "pChEMBL": p_val,
                    "90% Prediction Interval": interval_str,
                    "Interval Width": width,
                    "Uncertainty (std equiv)": unc_val,
                    "Hit": k in r["target_hits"]
                })
            st.table(rows)
            
            # Export results table as CSV
            pred_df = pd.DataFrame(rows)
            pred_csv = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions (CSV)",
                data=pred_csv,
                file_name="adenosine_predictions.csv",
                mime="text/csv",
                key="dl_single_predictions_csv"
            )

            if r["target_hits"]:
                st.write("**Targets above threshold:**", ", ".join(r["target_hits"]))
            else:
                st.write("**No targets met the standard active threshold (pChEMBL ≥ 6.0).**")

            # 6. Live Local SHAP explanation
            if r["source"] == "model":
                st.subheader("SHAP Feature Attribution (Explainability)")
                st.write(f"Local feature contributions for the predicted **{r['best_target']}** affinity:")
                try:
                    from src.features import build_features
                    
                    # Load conformal model
                    model_path = Path(f"models/precise/xgboost_precise_{r['best_target'].lower()}_model.pkl")
                    if not model_path.exists():
                        model_path = Path(f"models/xgboost_{r['best_target'].lower()}_model.pkl")
                    
                    with open(model_path, "rb") as f_model:
                        model_conformal = pickle.load(f_model)
                    
                    # Extract base estimator
                    if type(model_conformal).__name__ == "CrossConformalRegressor":
                        estimator = model_conformal._mapie_regressor.estimator_.estimators_[0]
                    elif type(model_conformal).__name__ == "MapieRegressor":
                        estimator = model_conformal.estimators_[0]
                    elif isinstance(model_conformal, list) and len(model_conformal) > 0:
                        estimator = model_conformal[0]
                    else:
                        estimator = model_conformal
                    
                    # Load scaler
                    with open("models/scaler.pkl", "rb") as f_scaler:
                        pipeline = pickle.load(f_scaler)
                    
                    # Reconstruct feature names
                    feature_names = [f"Morgan_FP_{i}" for i in range(2048)] + [f"MACCS_{i}" for i in range(167)]
                    selected_desc_names = pipeline.feature_filter.feature_names
                    feature_names.extend(selected_desc_names)
                    
                    # Generate x
                    x = build_features(canon, pipeline).reshape(1, -1)
                    
                    # Compute local SHAP with feature names mapped properly
                    X_df = pd.DataFrame(x, columns=feature_names)
                    explainer = shap.TreeExplainer(estimator)
                    shap_values = explainer(X_df)
                    
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
                    plt.title(f"SHAP Local Contribution Breakdown: {r['best_target']}", fontsize=11, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Chemical explanation for top contributors
                    import numpy as np
                    sv = shap_values[0]
                    top_indices = np.argsort(np.abs(sv.values))[::-1][:5]
                    
                    st.markdown("#### 🔬 Medicinal Chemistry Translation of Contributing Features")
                    st.write("Interpretation of the most significant descriptors driving this specific prediction:")
                    
                    explanations = []
                    for idx in top_indices:
                        f_name = feature_names[idx]
                        f_val = sv.data[idx]
                        shap_val = sv.values[idx]
                        
                        exp = explain_feature_chemically(f_name, f_val, smiles)
                        if exp:
                            direction = "📈 Increases Affinity" if shap_val > 0 else "📉 Decreases Affinity"
                            explanations.append({
                                "Feature": f_name,
                                "Feature Type": exp["Type"],
                                "Value in Query": exp["Query Value"],
                                "Impact on Affinity": f"{direction} (SHAP = {shap_val:+.3f})",
                                "Chemical Interpretation / Environment Map": exp["Interpretation"]
                            })
                            
                    if explanations:
                        st.dataframe(pd.DataFrame(explanations), use_container_width=True)
                except Exception as e:
                    st.caption(f"SHAP visual attribution details: {e}")

            # 7. Direct Selectivity Profile (ΔpChEMBL)
            if "selectivity_profile" in r and r["selectivity_profile"]:
                st.subheader("Direct Selectivity Profile (ΔpChEMBL)")
                st.write("Direct pairwise selectivity estimates utilizing delta-affinity models:")
                sel_rows = []
                for pair, diff in r["selectivity_profile"].items():
                    subA, subB = pair.split("_vs_")
                    preferred = subA if diff > 0 else subB
                    fold_ratio = round(10 ** abs(diff), 1)
                    
                    sel_rows.append({
                        "Comparison": f"{subA} vs {subB}",
                        "Predicted ΔpChEMBL": diff,
                        "Preferred Subtype": preferred,
                        "Affinity Ratio (Fold Preference)": f"{fold_ratio}x"
                    })
                st.table(sel_rows)

            # 8. Reliability / Applicability Domain
            st.subheader("Reliability / Applicability Domain")
            sim = nearest_tanimoto(smiles)
            if sim is None:
                st.warning("AD cache missing. Run the feature pipeline to generate data/processed/train_fps.pkl.")
            else:
                label = _ad_label(sim)
                st.metric("Nearest Tanimoto (train)", f"{sim:.3f}")
                if label == "High":
                    st.success(f"Reliability: {label} (≥ 0.6 – well within training domain).")
                elif label == "Medium":
                    st.warning(f"Reliability: {label} (0.4 – 0.6 – moderate confidence).")
                else:
                    st.error(f"Reliability: {label} (< 0.4 – out-of-domain; use with caution).")

            # 9. Safety & Drug-Likeness Profiles
            st.subheader("Safety & Drug-Likeness")
            col_pains, col_qed = st.columns(2)
            
            with col_pains:
                alerts = check_pains(smiles)
                if alerts:
                    st.error(f"PAINS alert(s) detected: {', '.join(alerts)}")
                else:
                    st.success("No PAINS structural alerts detected.")

            with col_qed:
                if profile:
                    st.metric("QED Score", f"{qed_val:.3f}")
                else:
                    st.warning("Could not compute drug-likeness.")

            # Top-5 Similar Training Molecules - Only if NOT in database
            st.subheader("Top-5 Similar Training Molecules (Tanimoto, Morgan r=2)")
            
            try:
                # Get the canonical smiles and top similarities
                canon_smi, top_sims = topk_tanimoto(smiles, k=5)
                
                if canon_smi is None:
                    st.write("No similarity results (invalid SMILES).")
                elif not top_sims:
                    st.error(
                        "Similarity cache missing. Run the feature pipeline once to generate "
                        "data/processed/train_fps.pkl and data/processed/train_smiles.pkl."
                    )
                else:
                    st.markdown(f"**Canonical SMILES Query:** `{canon_smi}`")
                    
                    # Display the table
                    sim_rows = [{"Train SMILES": s, "Tanimoto": round(sim, 4)} for s, sim in top_sims]
                    st.table(sim_rows)
            except Exception as e:
                st.error(f"Similarity search failed: {e}")

def _section_batch_prediction():
    st.header("Batch CSV Prediction")

    uploaded = st.file_uploader("Upload a CSV", type="csv")
    if uploaded is None:
        st.info("Upload a CSV with a SMILES column to begin.")
        return

    df = pd.read_csv(uploaded)
    # Using the helper from the component to stay consistent
    from src.app.components.batch_predict import _infer_smiles_col 
    smiles_col = _infer_smiles_col(df)
    st.write(f"Detected SMILES column: **{smiles_col}** | Total Rows: {len(df)}")

    # Model mode is set to unified precise model
    mode = "precise"
    
    threshold = 6.0

    if st.button("Run Batch Prediction"):
        with st.spinner("Processing..."):
            result_df = predict_batch(df, threshold=threshold, smiles_col=smiles_col, mode=mode)

        # Check for errors column securely
        if "error" in result_df.columns:
            err_count = result_df["error"].notna().sum()
            if err_count > 0:
                st.warning(f"Processed {len(result_df)} rows; {err_count} invalid SMILES skipped.")
        
        # Display the 4 independent subtype results
        display_cols = [smiles_col, 'A1', 'A2A', 'A2B', 'A3', 'best_target', 'in_database']
        existing = [c for c in display_cols if c in result_df.columns]
        st.dataframe(result_df[existing], use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", data=csv, file_name="ar_batch_results.csv")

def _section_results():
    st.header("Model Validation & Diagnostics Results")
    
    view_mode = "precise"
    
    tab_metrics, tab_shap_y, tab_a1_diag, tab_examples = st.tabs([
        "Validation Metrics", "TreeSHAP & Y-Randomization", "Dataset Quality Diagnostics", "Example Predictions"
    ])
  
    with tab_metrics:
        # Check if we have Nested CV report to show
        nested_cv_report = Path("outputs/nested_cv/merged_report.md")
        if nested_cv_report.exists():
            st.subheader("Deterministic Nested Cross-Validation (Scaffold Split + HPO)")
            st.write("Aggregated 5-fold outer scaffold scaffold-split performance with Optuna hyperparameter optimization in the inner loop (laptop-safe sequential chunks):")
            with open(nested_cv_report, "r") as f_ncv:
                st.markdown(f_ncv.read())
            st.divider()
            
        try:
            base_dir = "outputs/validoutput/precise"
            overall_df, per_df = load_evaluation_tables(base_dir)
            if not overall_df.empty:
                st.subheader("Ensemble Metrics vs Baseline (PRECISE Mode)")
                st.dataframe(overall_df, use_container_width=True)
            if not per_df.empty:
                st.subheader("Per-Receptor Subtype Metrics (PRECISE Mode)")
                st.dataframe(per_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load evaluation report: {e}")
  
        img_path = "outputs/validoutput/precise/calibration_precise_plot.png"
        if not Path(img_path).exists():
            img_path = "outputs/validoutput/precise/calibration_root_plot.png"
        if not Path(img_path).exists():
            img_path = "outputs/calibration_plot.png"
        if Path(img_path).exists():
            st.subheader("Calibration Plot (PRECISE Mode)")
            st.image(img_path, use_container_width=True)
  
    with tab_a1_diag:
        st.subheader("Dataset Quality & Activity Cliff Diagnostics")
        st.write("Curated dataset analysis of adenosine receptor subtypes identifying structural bottlenecks:")
        
        diag_choice = st.selectbox(
            "Select Diagnostic Target", 
            ["Combined Overview", "A1 Subtype", "A2A Subtype", "A2B Subtype", "A3 Subtype"],
            key="diagnostics_selector"
        )
        
        if diag_choice == "Combined Overview":
            diag_report_path = Path("outputs/diagnostics/combined_diagnosis_report.json")
            if diag_report_path.exists():
                with open(diag_report_path, "r") as f_diag:
                    diag_data = json.load(f_diag)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Curated Compounds", diag_data["n_compounds"])
                col2.metric("Unique Murcko Scaffolds", diag_data["scaffold_diversity"]["n_unique_scaffolds"])
                col3.metric("Combined Scaffold Ratio", f"{diag_data['scaffold_diversity']['diversity_ratio']:.3f}")
                
                # Show subtype breakdown table
                st.markdown("### 📊 Subtype Compound Distribution")
                sb = diag_data["target_subtype_breakdown"]
                sb_df = pd.DataFrame([{"Receptor Subtype": k, "Compound Count": v, "Percentage": f"{v/diag_data['n_compounds']*100:.1f}%"} for k, v in sb.items()])
                st.dataframe(sb_df, use_container_width=True)
                
                # Combined pChEMBL distribution plot
                dist_plot = Path("outputs/diagnostics/combined_pchembl_distribution.png")
                if dist_plot.exists():
                    st.image(str(dist_plot), caption="Combined pChEMBL Affinity Distribution Profile", use_container_width=True)
                
                # Standard type breakdown
                st.markdown("### 🧬 Experimental Assay Measurement Types")
                tb = diag_data["standard_type_breakdown"]
                tb_df = pd.DataFrame([{"Measurement (Standard Type)": k, "Count": v, "Percentage": f"{v/diag_data['n_compounds']*100:.1f}%"} for k, v in tb.items()])
                st.dataframe(tb_df, use_container_width=True)
                
                # Model-diagnostics insight
                st.markdown("""
                > **💡 Cheminformatics Insight & Model Performance Link:**
                > 
                > Subtype **A2B** features the highest prediction accuracy ($R^2 = 0.632$) despite a moderate sample size because it contains highly structured canonical groups with minimal activity cliffs.
                > Conversely, subtype **A1** has a lower accuracy ($R^2 = 0.243$) because it has the highest ratio of activity cliffs and scaffold diversity, representing a challenging structural landscape.
                """)
            else:
                st.info("Combined diagnostics report not generated yet. Run the pipeline to populate.")
                
        else:
            # Map choice to subtype
            subtype_map = {
                "A1 Subtype": "a1",
                "A2A Subtype": "a2a",
                "A2B Subtype": "a2b",
                "A3 Subtype": "a3"
            }
            subtype_prefix = subtype_map[diag_choice]
            subtype_name = diag_choice.split(" ")[0] # e.g. "A1"
            
            diag_report_path = Path(f"outputs/diagnostics/{subtype_prefix}_diagnosis_report.json")
            if diag_report_path.exists():
                with open(diag_report_path, "r") as f_diag:
                    diag_data = json.load(f_diag)
                    
                col_sub_1, col_sub_2 = st.columns(2)
                with col_sub_1:
                    st.metric(f"Total {subtype_name} Compounds", diag_data["n_compounds"])
                    st.metric("Unique Scaffolds", diag_data["scaffold_diversity"]["n_unique_scaffolds"])
                    st.metric("Scaffold Diversity Ratio", f"{diag_data['scaffold_diversity']['diversity_ratio']:.3f}")
                with col_sub_2:
                    st.metric("Activity Cliffs Detected", diag_data["n_activity_cliffs"])
                    mean_val = diag_data["pchembl_stats"]["mean"]
                    std_val = diag_data["pchembl_stats"]["std"]
                    st.metric("pChEMBL Range", f"{mean_val:.2f} ± {std_val:.2f}")
                    
                # Display plots if they exist
                dist_plot = Path(f"outputs/diagnostics/{subtype_prefix}_pchembl_distribution.png")
                if dist_plot.exists():
                    st.image(str(dist_plot), caption=f"{subtype_name} pChEMBL Distribution Profile", use_container_width=True)
                    
                cliff_plot = Path(f"outputs/diagnostics/{subtype_prefix}_activity_cliffs_shifts.png")
                if cliff_plot.exists():
                    st.image(str(cliff_plot), caption=f"{subtype_name} Activity Cliff Magnitude Distribution", use_container_width=True)
                    
                if diag_data["activity_cliffs"]:
                    st.subheader(f"Top Detected Activity Cliffs in {subtype_name} (Tanimoto Similarity ≥ 0.8, |ΔpChEMBL| ≥ 1.5)")
                    cliff_df = pd.DataFrame(diag_data["activity_cliffs"])
                    st.dataframe(cliff_df[["tanimoto_similarity", "pchembl_difference", "compound_1_pchembl", "compound_2_pchembl"]], use_container_width=True)
            else:
                st.info(f"{subtype_name} Diagnostics report not generated yet. Run the pipeline to populate.")
            
    with tab_shap_y:
        st.subheader("TreeSHAP Global Explainability & Y-Randomization Validation")
        st.write("Chemical sanity checks and label-shuffling tests to verify structural target relationship:")
        
        col_shap, col_yrand = st.columns(2)
        with col_shap:
            st.markdown("### Global TreeSHAP Feature Importance (A2A)")
            st.write("Attribution of continuous descriptors and structural fingerprint features:")
            shap_bar = Path("outputs/shap/A2A_bar.png")
            shap_beeswarm = Path("outputs/shap/A2A_beeswarm.png")
            
            if shap_bar.exists():
                st.image(str(shap_bar), caption="SHAP Global Mean Absolute Attribution", use_container_width=True)
            if shap_beeswarm.exists():
                st.image(str(shap_beeswarm), caption="SHAP Beeswarm Distribution Plot", use_container_width=True)
                
            shap_report_path = Path("outputs/shap/A2A_shap_report.json")
            if shap_report_path.exists():
                with open(shap_report_path, "r") as f_shap:
                    shap_data = json.load(f_shap)
                sanity = shap_data["sanity_check"]
                st.markdown(f"**Chemical Sanity Check: `{sanity['status']}`**")
                st.write(sanity["message"])
                st.write(f"Key expected descriptors identified: `{', '.join(sanity['expected_features_found'])}`")
            else:
                st.info("SHAP explainability plots not generated yet. Run the pipeline.")
                
        with col_yrand:
            st.markdown("### Y-Randomization Validation (A2A)")
            st.write("Shuffling pChEMBL labels to ensure models don't overfit to noise or spurious background features:")
            
            yrand_plot = Path("outputs/y_randomization/A2A_distribution.png")
            if yrand_plot.exists():
                st.image(str(yrand_plot), caption="Y-Randomization Label-Shuffled R² Distribution", use_container_width=True)
                
            yrand_report_path = Path("outputs/y_randomization/A2A_report.json")
            if yrand_report_path.exists():
                with open(yrand_report_path, "r") as f_yrand:
                    yrand_data = json.load(f_yrand)
                st.metric("Real Model R² Score", f"{yrand_data['real_r2']:.3f}")
                st.metric("Label-Shuffled R² Score", f"{yrand_data['shuffled_r2_mean']:.3f} ± {yrand_data['shuffled_r2_std']:.3f}")
                if yrand_data["leakage_warning"]:
                    st.error("WARNING: Shuffled R² is high! Spurious target leakage detected.")
                else:
                    st.success("SUCCESS: Shuffled R² is near-zero. Model represents true chemical target SAR.")
            else:
                st.info("Y-Randomization statistics not generated yet. Run the pipeline.")
 
    with tab_examples:
        try:
            st.subheader("Run Summary (PRECISE Mode)")
            st.dataframe(load_run_summary(base_dir), use_container_width=True)
            db_df, novel_df = load_examples(base_dir)
            st.subheader("DB Hit Examples (PRECISE Mode)")
            st.dataframe(db_df, use_container_width=True)
            st.subheader("Novel Molecule Examples (PRECISE Mode)")
            st.dataframe(novel_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load examples: {e}")
 
def run_app():
    st.set_page_config(page_title="AR Selectivity Predictor", layout="wide")
    st.title("Adenosine Receptor Selectivity Predictor")

    # Prominent Scientific Goal & Trust card to explain webapp's purpose and back it with accuracy numbers
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 5px solid #005a9c; margin-bottom: 25px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <h3 style="margin-top: 0; color: #005a9c; font-size: 1.3rem;">Scientific Goal & Platform Trustworthiness</h3>
        <p style="font-size: 0.95rem; line-height: 1.5; color: #333333; margin-bottom: 12px;">
            This web application facilitates rapid, high-confidence in silico profiling of binding affinities (pChEMBL values) across all four adenosine receptor subtypes (A<sub>1</sub>, A<sub>2A</sub>, A<sub>2B</sub>, A<sub>3</sub>). By predicting candidate compound profiles, researchers can systematically identify subtype-selective hits and assess off-target liabilities early in the drug discovery process.
        </p>
        <div style="display: flex; flex-direction: row; gap: 15px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 320px; background: white; padding: 12px; border-radius: 6px; border: 1px solid #e9ecef;">
                <strong style="color: #005a9c;">🎯 Conformal Model Performance Metrics</strong>
                <ul style="margin: 5px 0 0 18px; padding: 0; font-size: 0.85rem; color: #555555;">
                    <li>Overall Ensemble Model <strong>MAE = 0.520 pChEMBL units</strong></li>
                    <li>Overall Ensemble Model <strong>R² = 0.408</strong></li>
                    <li>Validated on 9,589 parent compounds across all subtypes</li>
                </ul>
            </div>
            <div style="flex: 1; min-width: 320px; background: white; padding: 12px; border-radius: 6px; border: 1px solid #e9ecef;">
                <strong style="color: #005a9c;">🛡️ Conformal Prediction (MAPIE)</strong>
                <ul style="margin: 5px 0 0 18px; padding: 0; font-size: 0.85rem; color: #555555;">
                    <li>Dynamically outputs <strong>90% confidence prediction intervals</strong> for safety bounds</li>
                    <li>Strictly validated using Nested Cross-Validation with hyperparameter optimization</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_single, tab_batch, tab_results = st.tabs(["Single Prediction", "Batch Prediction", "Results"])

    with tab_single:
        _section_single_prediction()

    with tab_batch:
        _section_batch_prediction()
    with tab_results:
        _section_results()

if __name__ == "__main__":
    run_app()
