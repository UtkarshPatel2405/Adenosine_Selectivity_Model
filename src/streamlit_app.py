from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.predictor import predict
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

    smiles = st.text_input("SMILES", value="CCn1c(/N=C/c2ccc(Br)cc2)c(C#N)sc1=S")
    threshold = st.slider("Hit threshold (pChEMBL)", 4.0, 9.0, 6.0, 0.1)

    if st.button("Predict"):
       
        img = draw_2d(smiles)
        if img is not None:
            st.image(img, caption="2D Structure", width=350)
        else:
            st.warning("Could not render 2D structure – is the SMILES valid?")

       
        try:
            r = predict(smiles, threshold=threshold)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        if r["source"] == "database":
            st.info("Experimental data from ChEMBL (database hit).")
        else:
            st.info("ML model prediction.")

        st.write(f"**Best target:** {r['best_target']}")

        sel = r.get("selectivity_score")
        if sel is None:
            st.write("Selectivity score: N/A (only one subtype has experimental data).")
        else:
            sel = float(sel)
            st.write(f"Selectivity score: {sel:.3f}")
            if sel < 0.3:
                st.warning("Selectivity: low (likely non-selective across subtypes).")
            elif sel < 1.0:
                st.info("Selectivity: moderate.")
            else:
                st.success("Selectivity: high.")

        preds = r["predictions"]
        unc = r["uncertainty"]
        rows = [
            {
                "Subtype": k,
                "pChEMBL": None if preds.get(k) is None else round(float(preds[k]), 3),
                "Uncertainty (std)": None if unc.get(k) is None else round(float(unc[k]), 3),
                "Hit": preds.get(k) is not None and preds[k] > threshold,
            }
            for k in ["A1", "A2A", "A2B", "A3"]
        ]
        st.table(rows)

        hits = r.get("target_hits", [])
        if hits:
            st.write("Targets above threshold:", ", ".join(hits))
        else:
            st.write("No targets above threshold.")


        st.subheader("Reliability / Applicability Domain")
        sim = nearest_tanimoto(smiles)
        if sim is None:
            st.warning(
                "AD cache missing.  Run the feature pipeline to generate "
                "data/processed/train_fps.pkl."
            )
        else:
            label = _ad_label(sim)
            st.metric("Nearest Tanimoto (train)", f"{sim:.3f}")
            if label == "High":
                st.success(f"Reliability: {label} (≥ 0.6 – well within training domain).")
            elif label == "Medium":
                st.warning(f"Reliability: {label} (0.4 – 0.6 – moderate confidence).")
            else:
                st.error(f"Reliability: {label} (< 0.4 – out-of-domain; use with caution).")


        st.subheader("PAINS Alerts")
        alerts = check_pains(smiles)
        if alerts:
            st.error(f"PAINS alert(s) detected: {', '.join(alerts)}")
        else:
            st.success("No PAINS alerts detected.")

    
        st.subheader("Drug-Likeness Profile")
        profile = qed_profile(smiles)
        if profile is None:
            st.warning("Could not compute drug-likeness (invalid SMILES).")
        else:
            cols = st.columns(4)
            for i, (key, val) in enumerate(profile.items()):
                cols[i % 4].metric(key, val)

       
        st.subheader("Top-5 Similar Training Molecules (Tanimoto, Morgan r=2)")
        try:
            canon, top_sims = topk_tanimoto(smiles, k=5)
            if canon is None:
                st.write("No similarity results (invalid SMILES).")
            elif not top_sims:
                st.error(
                    "Similarity cache missing.  Run the feature pipeline once to generate "
                    "data/processed/train_fps.pkl and data/processed/train_smiles.pkl."
                )
            else:
                st.caption(f"Canonical SMILES used: {canon}")
                st.table([{"Train SMILES": s, "Tanimoto": round(sim, 4)} for s, sim in top_sims])
        except FileNotFoundError:
            st.error(
                "Similarity cache missing.  Run the feature pipeline once to generate "
                "data/processed/train_fps.pkl and data/processed/train_smiles.pkl."
            )


def _section_batch_prediction():
    st.header("Batch CSV Prediction")

    uploaded = st.file_uploader("Upload a CSV with a SMILES column", type="csv")
    if uploaded is None:
        st.info(
            "Upload a CSV to get started.  The file should contain a column named "
            "`smiles`, `SMILES`, or similar."
        )
        return

    df = pd.read_csv(uploaded)
    smiles_col = _infer_smiles_col(df)
    st.write(f"Detected SMILES column: **{smiles_col}** | Rows: {len(df)}")

    threshold = st.slider(
        "Hit threshold (pChEMBL) – batch", 4.0, 9.0, 6.0, 0.1, key="batch_thr"
    )

    if st.button("Run Batch Prediction"):
        with st.spinner("Running predictions…"):
            result_df = predict_batch(df, threshold=threshold, smiles_col=smiles_col)

        errors = result_df["error"].notna().sum()
        st.success(f"Done. {len(result_df)} rows processed; {errors} error(s).")
        st.dataframe(result_df, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results CSV",
            data=csv_bytes,
            file_name="batch_predictions.csv",
            mime="text/csv",
        )


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
 
        if Path("outputs/calibration_plot.png").exists():
            st.subheader("Calibration Plot")
            st.image("outputs/calibration_plot.png", use_container_width=True)
 
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
