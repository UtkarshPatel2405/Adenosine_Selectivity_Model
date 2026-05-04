from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.predictor import SUBTYPES, predict
from src.chem_utils import topk_tanimoto
from src.app.components.structure_viz import draw_2d
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
def _ad_label(sim: float | None) -> str:
    if sim is None:
        return "Unknown"
    if sim >= 0.6:
        return "High"
    if sim >= 0.4:
        return "Medium"
    return "Low"

def _section_single_prediction():
    st.header("Single SMILES Prediction")

    # User Inputs
    smiles = st.text_input("SMILES", value="CCn1c(/N=C/c2ccc(Br)cc2)c(C#N)sc1=S")
    threshold = st.slider("Hit threshold (pChEMBL)", 4.0, 9.0, 6.0, 0.1)

    if st.button("Predict"):
        # 1. Visualization
        img = draw_2d(smiles)
        if img is not None:
            st.image(img, caption="2D Structure", width=350)
        else:
            st.warning("Could not render 2D structure – is the SMILES valid?")

        # 2. Run Prediction Pipeline
        try:
            r = predict(smiles, threshold=threshold)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        # 3. Data Source Information[cite: 19]
        if r["in_database"]:
            st.success("Experimental data retrieved from ChEMBL (Database Hit).")
            st.caption("Note: Missing experimental values for specific subtypes are assumed as 0.000 per experimental protocol.")
        else:
            st.info("ML Ensemble model prediction (Novel Molecule).")
       
       
        st.subheader("Physicochemical Profile")
        d = r["descriptors"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mol. Weight", d["MW"])
        c2.metric("LogP", d["LogP"])
        c3.metric("H-Bond Donors", d["HBD"])
        c4.metric("H-Bond Acceptors", d["HBA"])
        
        c5, c6, c7, _ = st.columns(4)
        c5.metric("Rotatable Bonds", d["RotBonds"])
        c6.metric("Aromatic Rings", d["AromRings"])
        c7.metric("TPSA", d["TPSA"])

        st.subheader("Subtype Bioactivity Profile")
        st.write(f"**Primary Target Receptor:** {r['best_target']}")
        preds, unc = r["predictions"], r["uncertainty"]
        
        rows = [{
            "Subtype": k,
            "pChEMBL": round(float(preds[k]), 3),
            "Uncertainty (std)": round(float(unc[k]), 3),
            "Hit": k in r["target_hits"]
        } for k in SUBTYPES]
        st.table(rows)

        if r["target_hits"]:
            st.write("**Targets above threshold:**", ", ".join(r["target_hits"]))
        else:
            st.write("**No targets met the current pChEMBL threshold.**")

        # 6. Reliability / Applicability Domain[cite: 21]
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

        # 7. Safety & Drug-Likeness Profiles[cite: 21]
        st.subheader("Safety & Drug-Likeness")
        col_pains, col_qed = st.columns(2)
        
        with col_pains:
            alerts = check_pains(smiles)
            if alerts:
                st.error(f"PAINS alert(s) detected: {', '.join(alerts)}")
            else:
                st.success("No PAINS structural alerts detected.")

        with col_qed:
            profile = qed_profile(smiles)
            if profile:
                st.metric("QED Score", f"{profile.get('QED', 0.0):.3f}")
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
                        # THIS LINE uses the canon_smi variable so it won't be dull!
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

    threshold = st.slider("pChEMBL Hit Threshold", 4.0, 9.0, 6.0, 0.1)

    if st.button("Run Batch Prediction"):
        with st.spinner("Processing..."):
            result_df = predict_batch(df, threshold=threshold, smiles_col=smiles_col)

        # Check for errors column securely[cite: 18]
        if "error" in result_df.columns:
            err_count = result_df["error"].notna().sum()
            if err_count > 0:
                st.warning(f"Processed {len(result_df)} rows; {err_count} invalid SMILES skipped.")
        
        # Display the 4 independent subtype results[cite: 19]
        display_cols = [smiles_col, 'A1', 'A2A', 'A2B', 'A3', 'best_target', 'in_database']
        existing = [c for c in display_cols if c in result_df.columns]
        st.dataframe(result_df[existing], use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", data=csv, file_name="ar_batch_results.csv")

def _section_results():
    st.header("Model Results")
 
    # Check which files exist and warn if missing
    status = outputs_exist()
    missing = [f for f, exists in status.items() if not exists]
    if missing:
        st.warning(f"Missing output files: {', '.join(missing)}")
 
    tab_metrics, tab_ood, tab_fp, tab_examples, tab_modes = st.tabs([
        "Metrics", "Scaffold OOD", "Fingerprint Comparison", "Examples", "Standard vs Strict"
    ])
 
    with tab_metrics:
        try:
            overall_df, per_df = load_evaluation_tables()
            st.subheader("Overall Metrics vs Baseline")
            st.dataframe(overall_df, use_container_width=True)
            st.subheader("Per-Subtype Metrics")
            st.dataframe(per_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load evaluation report: {e}")
 
        img_path = "outputs/validoutput/standard/calibration_plot.png"
        if Path(img_path).exists():
            st.subheader("Calibration Plot standard")
            st.image(img_path, use_container_width=True)

        img_path = "outputs/validoutput/strict/calibration_plot.png"
        if Path(img_path).exists():
            st.subheader("Calibration Plot strict")
            st.image(img_path, use_container_width=True)
 
    with tab_ood:
        try:
            st.subheader("Scaffold OOD Report")
            st.dataframe(load_scaffold_ood(), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load scaffold OOD report: {e}")
 
    with tab_fp:
        try:
            st.subheader("Fingerprint Comparison")
            st.dataframe(load_fingerprint_comparison(), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load fingerprint comparison: {e}")
 
    with tab_examples:
        try:
            st.subheader("Run Summary")
            st.dataframe(load_run_summary(), use_container_width=True)
            db_df, novel_df = load_examples()
            st.subheader("DB Hit Examples")
            st.dataframe(db_df, use_container_width=True)
            st.subheader("Novel Molecule Examples")
            st.dataframe(novel_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load examples: {e}")
 
    with tab_modes:
        st.subheader("Standard vs Strict Mode Comparison")
        col_std, col_strict = st.columns(2)
        for col, mode in zip([col_std, col_strict], ["standard", "strict"]):
            with col:
                st.markdown(f"**{mode.title()} mode**")
                try:
                    summary_df, db_df, novel_df = load_mode_examples(mode)
                    st.caption("Run summary")
                    st.dataframe(summary_df, use_container_width=True)
                    st.caption("DB examples")
                    st.dataframe(db_df, use_container_width=True)
                    st.caption("Novel examples")
                    st.dataframe(novel_df, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load {mode} outputs: {e}")
 
 
    
        
def run_app():
    st.set_page_config(page_title="AR Selectivity Predictor", layout="wide")
    st.title("Adenosine Receptor Selectivity Predictor")

    tab_single, tab_batch, tab_results = st.tabs(["Single Prediction", "Batch Prediction", "Results"])

    with tab_single:
        _section_single_prediction()

    with tab_batch:
        _section_batch_prediction()
    with tab_results:
        _section_results()
