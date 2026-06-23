from __future__ import annotations
import json, sys, pickle, numpy as np
from pathlib import Path
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predictor import SUBTYPES, predict
from src.chem_utils import topk_tanimoto
from src.app.components.structure_viz import draw_2d_svg, generate_3d_conformer, generate_pdb_block
from src.app.components.pains_checker import check_pains
from src.app.components.drug_likeness import qed_profile
from src.app.components.applicability_domain import nearest_tanimoto
from src.app.components.batch_predict import predict_batch, _infer_smiles_col
from src.app.components.model_reports import (
    load_evaluation_tables, load_run_summary, load_examples, _load_json, outputs_exist,
)

# ── Caching ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_predict(smiles: str, threshold: float) -> dict:
    return predict(smiles, threshold)

@st.cache_resource(show_spinner=False)
def _load_shap_model(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

# ── CSS ──────────────────────────────────────────────────────────────────────
_MAIN_CSS = """
<style>
/* ─── Variables ─── */
:root {
  --navy: #0a1e3d;
  --navy-light: #1a3a6a;
  --teal: #00b4d8;
  --teal-dark: #0096b4;
  --green: #2ecc71;
  --amber: #f39c12;
  --red: #e74c3c;
  --bg: #f4f6f9;
  --card: #ffffff;
  --border: #e2e6ed;
  --text: #1a2330;
  --text-secondary: #5a6a7a;
}
/* ─── Global ─── */
.main > div { padding: 0.5rem 1rem; }
.stApp { background: var(--bg); }
h1, h2, h3, h4 { font-family: 'Inter', -apple-system, sans-serif !important; letter-spacing: -0.01em; }
p, li, div, span { font-family: 'Inter', -apple-system, sans-serif; }
/* ─── Header Hero ─── */
.hero-card {
  background: linear-gradient(135deg, #0a1e3d 0%, #1a3a6a 100%);
  border-radius: 16px; padding: 1.8rem 2rem; margin-bottom: 1.5rem;
  color: white; position: relative; overflow: hidden;
}
.hero-card::after {
  content: ''; position: absolute; top: -50%; right: -20%;
  width: 400px; height: 400px; background: radial-gradient(circle, rgba(0,180,216,0.12) 0%, transparent 70%);
  border-radius: 50%;
}
.hero-title { font-size: 1.6rem; font-weight: 700; margin: 0 0 0.4rem 0; letter-spacing: -0.02em; }
.hero-sub { font-size: 0.9rem; opacity: 0.8; max-width: 700px; line-height: 1.5; margin-bottom: 1rem; }
.hero-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; position: relative; z-index: 1; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 0.3rem;
  background: rgba(255,255,255,0.12); backdrop-filter: blur(4px);
  padding: 0.3rem 0.7rem; border-radius: 20px;
  font-size: 0.75rem; border: 1px solid rgba(255,255,255,0.15);
}
/* ─── Metrics Dashboard ─── */
.metrics-dash {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.6rem; margin-top: 1rem; position: relative; z-index: 1;
}
.metric-card {
  background: rgba(255,255,255,0.08); backdrop-filter: blur(4px);
  border: 1px solid rgba(255,255,255,0.12); border-radius: 10px;
  padding: 0.6rem 0.8rem; transition: all 0.2s;
}
.metric-card:hover { background: rgba(255,255,255,0.14); transform: translateY(-1px); }
.metric-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; margin-bottom: 0.1rem; }
.metric-value { font-size: 1.2rem; font-weight: 700; }
.metric-sub { font-size: 0.7rem; opacity: 0.6; }
/* ─── Content Cards ─── */
.card {
  background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.2rem; margin-bottom: 1rem; transition: box-shadow 0.2s;
}
.card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
.card-title { font-size: 0.85rem; font-weight: 600; color: var(--text); margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem; }
/* ─── Status Tags ─── */
.tag { display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.15rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 500; }
.tag-green { background: #e8f8f0; color: #1a7d4a; }
.tag-amber { background: #fef3e2; color: #b0700a; }
.tag-red { background: #fde8e8; color: #b91c1c; }
.tag-blue { background: #e0f0fe; color: #1a5a9c; }
/* ─── Metric Grid ─── */
.mgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.5rem; }
.mitem { background: var(--bg); border-radius: 8px; padding: 0.5rem 0.7rem; text-align: center; }
.mitem-val { font-size: 1.1rem; font-weight: 700; }
.mitem-lbl { font-size: 0.65rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.03em; }
/* ─── Prediction Bar ─── */
.pred-bar { height: 6px; border-radius: 3px; margin: 0.3rem 0; background: var(--border); position: relative; overflow: hidden; }
.pred-bar-fill { height: 100%; border-radius: 3px; transition: width 0.6s ease; }
/* ─── Section Divider ─── */
.section-divider { height: 1px; background: linear-gradient(to right, transparent, var(--border), transparent); margin: 1.2rem 0; }
/* ─── Input ─── */
.stTextInput>div>div>input { border-radius: 10px !important; border: 2px solid var(--border) !important; padding: 0.6rem 0.8rem !important; font-size: 0.9rem !important; }
.stTextInput>div>div>input:focus { border-color: var(--teal) !important; box-shadow: 0 0 0 3px rgba(0,180,216,0.15) !important; }
/* ─── Buttons ─── */
.stButton>button {
  background: linear-gradient(135deg, #1a3a6a 0%, #2a5a9a 100%) !important;
  color: white !important; border: none !important; border-radius: 10px !important;
  padding: 0.4rem 1.5rem !important; font-weight: 500 !important; font-size: 0.85rem !important;
  transition: all 0.2s !important;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(26,58,106,0.3) !important; }
/* ─── Styled Tabs ─── */
.stTabs [data-baseweb="tab-list"] { gap: 0; border-radius: 12px; background: var(--card); padding: 0.3rem; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; padding: 0.4rem 1rem !important; font-size: 0.8rem !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #1a3a6a 0%, #2a5a9a 100%) !important; color: white !important; }
</style>
"""

def _tag(val, threshold=6.0):
    p = float(val) if val is not None else 0
    if p >= threshold: return "tag-green", "↑ Active"
    if p >= threshold - 1.5: return "tag-amber", "→ Weak"
    return "tag-red", "↓ Inactive"

def _impact(val):
    if val is None: return "tag-blue", "N/A"
    if val >= 0.6: return "tag-green", "High"
    if val >= 0.4: return "tag-amber", "Medium"
    return "tag-red", "Low"

# ── Sections ─────────────────────────────────────────────────────────────────
def _section_single_prediction():
    EXAMPLE_SMILES = {
        "Adenosine (endogenous)": "C1=NC2=C(C(=N1)N)N=CN2[C@@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O",
        "Caffeine (antagonist)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Theophylline (bronchodilator)": "CN1C2=C(C(=O)N(C1=O)C)NC=N2",
        "Regadenoson (A2A agonist)": "C1=NC2=C(C(=N1)N)N=CN2[C@@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O",
        "Custom SMILES": "CCn1c(/N=C/c2ccc(Br)cc2)c(C#N)sc1=S",
    }

    col_inp, col_ex = st.columns([3, 1])
    with col_inp:
        smiles = st.text_input("SMILES Input", value="CCn1c(/N=C/c2ccc(Br)cc2)c(C#N)sc1=S", label_visibility="collapsed")
    with col_ex:
        example = st.selectbox("", list(EXAMPLE_SMILES.keys()), label_visibility="collapsed")
        if example and example != "Custom SMILES":
            smiles = EXAMPLE_SMILES[example]

    if st.button("🔬 Predict & Analyze", use_container_width=True):
        with st.spinner("Running ML ensemble predictions…"):
            mol_block, min_charge, max_charge = generate_3d_conformer(smiles)
            svg = draw_2d_svg(smiles)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ── Structure ──
            c2d, c3d = st.columns(2)
            with c2d:
                if svg:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">🧪 2D Structure</div>', unsafe_allow_html=True)
                    st.image(svg, use_container_width=True)
                    st.download_button("⬇ SVG", data=svg, file_name="structure.svg", mime="image/svg+xml")
                    st.markdown('</div>', unsafe_allow_html=True)
            with c3d:
                if mol_block:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">🧬 3D Conformer</div>', unsafe_allow_html=True)
                    st.components.v1.html(_render_3d(mol_block), height=340)
                    c1, c2 = st.columns(2)
                    with c1: st.download_button("⬇ SDF", data=mol_block, file_name="conf.sdf")
                    with c2:
                        pdb = generate_pdb_block(smiles)
                        if pdb: st.download_button("⬇ PDB", data=pdb, file_name="conf.pdb")
                    st.markdown('</div>', unsafe_allow_html=True)

            # ── Prediction ──
            try:
                r = _cached_predict(smiles, threshold=6.0)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return
            canon = r["smiles"]

            src_tag = "📦 Database hit" if r["in_database"] else "🧠 ML Ensemble prediction"
            st.markdown(f'<div style="margin-bottom:0.5rem;"><span class="tag tag-blue">{src_tag}</span></div>', unsafe_allow_html=True)

            # ── Physicochemical profile ──
            d = r["descriptors"]
            profile = qed_profile(smiles)
            qed_val = profile.get("QED", 0.0) if profile else 0.0

            st.markdown('<div class="card-title">📊 Physicochemical Properties</div>', unsafe_allow_html=True)
            st.markdown('<div class="mgrid">', unsafe_allow_html=True)
            props = [
                ("MW", f"{d['MW']}"), ("LogP", f"{d['LogP']}"), ("HBD", str(d['HBD'])), ("HBA", str(d['HBA'])),
                ("RotB", str(d['RotBonds'])), ("AromRng", str(d['AromRings'])), ("TPSA", f"{d['TPSA']}"), ("QED", f"{qed_val:.2f}"),
            ]
            for lbl, val in props:
                st.markdown(f'<div class="mitem"><div class="mitem-lbl">{lbl}</div><div class="mitem-val">{val}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Bioactivity Profile ──
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card-title">🎯 Predicted Affinity Profile <span class="tag tag-blue">Best: {r["best_target"]}</span></div>', unsafe_allow_html=True)

            preds, unc = r["predictions"], r["uncertainty"]
            for k in SUBTYPES:
                xgb_val = preds.get("XGBoost", {}).get(k, 0)
                cls, lbl = _tag(xgb_val)
                bar_pct = min(float(xgb_val) / 10 * 100, 100)
                st.markdown(f'''
                <div class="card" style="padding: 0.6rem 1rem;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-weight:600;font-size:0.9rem;">A<sub>{k[1:].lower() if len(k)>1 else k}</sub></span>
                    <span><span class="tag {cls}">{lbl}</span> <span style="font-weight:700;font-size:1rem;margin-left:0.5rem;">{xgb_val:.2f}</span></span>
                  </div>
                  <div class="pred-bar"><div class="pred-bar-fill" style="width:{bar_pct}%;background:linear-gradient(90deg,{'#2ecc71' if xgb_val>=6 else '#f39c12' if xgb_val>=4.5 else '#e74c3c'},{'#27ae60' if xgb_val>=6 else '#e67e22' if xgb_val>=4.5 else '#c0392b'});"></div></div>
                  <div style="font-size:0.7rem;color:var(--text-secondary);display:flex;justify-content:space-between;">
                    <span>XGBoost</span>
                    <span>Uncertainty: {unc.get("XGBoost",{}).get(k,0):.3f}</span>
                  </div>
                </div>
                ''', unsafe_allow_html=True)

            # ── SHAP ──
            if r["source"] == "model":
                with st.expander("🔬 SHAP Feature Attribution (Why this prediction?)", expanded=False):
                    try:
                        from src.features import build_features
                        mp = Path(f"models/precise/xgboost_{r['best_target']}_production.pkl")
                        if not mp.exists(): mp = Path(f"models/precise/xgboost_precise_{r['best_target'].lower()}_model.pkl")
                        if not mp.exists(): mp = Path(f"models/xgboost_{r['best_target'].lower()}_model.pkl")
                        model_c = _load_shap_model(mp)
                        et = model_c if type(model_c).__name__ not in ("CrossConformalRegressor", "MapieRegressor") else (
                            model_c._mapie_regressor.estimator_.estimators_[0] if type(model_c).__name__ == "CrossConformalRegressor" else model_c.estimators_[0]
                        )
                        sp = Path("models/precise/scaler_precise.pkl")
                        if not sp.exists(): sp = Path("models/scaler.pkl")
                        with open(sp, "rb") as f: pl = pickle.load(f)
                        fn = [f"Morgan_FP_{i}" for i in range(2048)] + [f"MACCS_{i}" for i in range(167)] + list(pl.feature_filter.feature_names)
                        x = build_features(canon, pl).reshape(1, -1)
                        X_df = pd.DataFrame(x, columns=fn)
                        exp = shap.TreeExplainer(et); sv = exp(X_df)
                        fig, ax = plt.subplots(figsize=(7, 3.5))
                        shap.plots.waterfall(sv[0], max_display=6, show=False)
                        plt.title(f"Local SHAP: {r['best_target']}", fontsize=10, fontweight="bold")
                        plt.tight_layout(); st.pyplot(fig); plt.close()
                    except Exception as e:
                        st.caption(f"SHAP unavailable: {e}")

            # ── Safety & Drug-likeness ──
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                alerts = check_pains(smiles)
                if alerts: st.markdown(f'<div class="card"><span class="card-title">⚠️ Safety</span><span class="tag tag-red">{len(alerts)} PAINS alerts</span></div>', unsafe_allow_html=True)
                else: st.markdown(f'<div class="card"><span class="card-title">✅ Safety</span><span class="tag tag-green">No PAINS alerts</span></div>', unsafe_allow_html=True)
            with col_s2:
                sim = nearest_tanimoto(smiles)
                if sim is not None:
                    icls, ilbl = _impact(sim)
                    st.markdown(f'<div class="card"><span class="card-title">🎯 Applicability Domain</span><span class="tag {icls}">{ilbl} ({sim:.3f})</span></div>', unsafe_allow_html=True)
            with col_s3:
                st.markdown(f'<div class="card"><span class="card-title">💊 Drug-likeness</span><span class="tag tag-blue">QED: {qed_val:.3f}</span></div>', unsafe_allow_html=True)

            # ── Top-5 similar ──
            with st.expander("🔗 Top-5 Similar Training Molecules"):
                try:
                    canon_smi, top_sims = topk_tanimoto(smiles, k=5)
                    if top_sims:
                        st.markdown(f"Query: `{canon_smi}`")
                        rows = [{"#": i+1, "SMILES": s, "Tanimoto": f"{t:.4f}"} for i, (s, t) in enumerate(top_sims)]
                        st.dataframe(pd.DataFrame(rows).set_index("#"), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.caption(f"Similarity search: {e}")

def _section_batch_prediction():
    st.markdown('<div class="card"><div class="card-title">📁 Batch CSV Prediction</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is None:
        st.info("Upload a CSV with a SMILES column to begin.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    df = pd.read_csv(uploaded)
    smiles_col = _infer_smiles_col(df)
    st.markdown(f'Detected SMILES column: **{smiles_col}** &nbsp;|&nbsp; Rows: {len(df)}')
    if st.button("🚀 Run Batch Prediction", use_container_width=True):
        with st.spinner("Processing..."):
            result_df = predict_batch(df, threshold=6.0, smiles_col=smiles_col, mode="precise")
        if "error" in result_df.columns:
            ec = result_df["error"].notna().sum()
            if ec > 0: st.warning(f"{ec} invalid SMILES skipped.")
        display_cols = [c for c in [smiles_col, 'A1', 'A2A', 'A2B', 'A3', 'best_target', 'in_database'] if c in result_df.columns]
        st.dataframe(result_df[display_cols], use_container_width=True)
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Results (CSV)", data=csv, file_name="batch_results.csv")
    st.markdown('</div>', unsafe_allow_html=True)

def _section_results():
    base_dir = "outputs/validoutput/precise"
    tabs = st.tabs(["📊 Validation Metrics", "🧩 SHAP & Y-Rand", "📋 Diagnostics", "🔍 Examples", "🧠 GNN Comparison", "🔬 External Val.", "📚 Literature"])

    with tabs[0]:
        try:
            overall, per = load_evaluation_tables(base_dir)
            if not overall.empty: st.markdown('<div class="card"><div class="card-title">📊 Overall Metrics</div>', unsafe_allow_html=True); st.dataframe(overall, use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
            if not per.empty: st.markdown('<div class="card"><div class="card-title">📊 Per-Subtype Metrics</div>', unsafe_allow_html=True); st.dataframe(per, use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e: st.warning(f"Could not load: {e}")
        for p in ["outputs/validoutput/precise/calibration_precise_plot.png", "outputs/validoutput/precise/calibration_root_plot.png", "outputs/calibration_plot.png"]:
            if Path(p).exists(): st.image(p, use_container_width=True); break

    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        sc = st.selectbox("Select subtype", SUBTYPES, key="shap_y")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**SHAP Global Importance**")
            for f in [Path(f"outputs/shap/{sc}_bar.png"), Path(f"outputs/shap/{sc}_beeswarm.png")]:
                if f.exists(): st.image(str(f), use_container_width=True)
        with c2:
            st.markdown("**Y-Randomization**")
            yrp = Path(f"outputs/y_randomization/{sc}_distribution.png")
            if yrp.exists(): st.image(str(yrp), use_container_width=True)
            yrr = Path(f"outputs/y_randomization/{sc}_report.json")
            if yrr.exists():
                yd = json.loads(yrr.read_text())
                st.metric("Real R²", f"{yd['real_r2']:.3f}")
                st.metric("Shuffled R²", f"{yd['shuffled_r2_mean']:.3f} ± {yd['shuffled_r2_std']:.3f}")
                if yd["leakage_warning"]: st.error("⚠ Possible leakage detected!")
                else: st.success("✅ Model represents true SAR")
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        dg = st.selectbox("Select target", ["Combined", "A1", "A2A", "A2B", "A3"], key="diag")
        if dg == "Combined":
            dp = Path("outputs/diagnostics/combined_diagnosis_report.json")
            if dp.exists():
                dd = json.loads(dp.read_text())
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Compounds", dd["n_compounds"]); c2.metric("Scaffolds", dd["scaffold_diversity"]["n_unique_scaffolds"]); c3.metric("Diversity", f"{dd['scaffold_diversity']['diversity_ratio']:.3f}")
                dist = Path("outputs/diagnostics/combined_pchembl_distribution.png")
                if dist.exists(): st.image(str(dist), use_container_width=True)
        else:
            pref = dg.lower()
            dp = Path(f"outputs/diagnostics/{pref}_diagnosis_report.json")
            if dp.exists():
                dd = json.loads(dp.read_text())
                c1, c2 = st.columns(2)
                with c1: st.metric(f"{dg} Compounds", dd["n_compounds"]); st.metric("Scaffolds", dd["scaffold_diversity"]["n_unique_scaffolds"])
                with c2: st.metric("Activity Cliffs", dd.get("n_activity_cliffs", 0)); st.metric("pChEMBL", f"{dd['pchembl_stats']['mean']:.2f} ± {dd['pchembl_stats']['std']:.2f}")
                for pfx in ["pchembl_distribution", "activity_cliffs_shifts"]:
                    f = Path(f"outputs/diagnostics/{pref}_{pfx}.png")
                    if f.exists(): st.image(str(f), use_container_width=True)

    with tabs[3]:
        try:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.dataframe(load_run_summary(base_dir), use_container_width=True)
            db_df, novel_df = load_examples(base_dir)
            if not db_df.empty: st.markdown("**Database Examples**"); st.dataframe(db_df, use_container_width=True)
            if not novel_df.empty: st.markdown("**Novel Molecule Examples**"); st.dataframe(novel_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e: st.warning(f"Examples: {e}")

    with tabs[4]:
        ep = Path("outputs/validoutput/precise/evaluation_precise_report.json")
        if ep.exists():
            try:
                ed = _load_json(str(ep)) or {}
                rows = [{"Subtype": s,
                    "XGBoost R²": ed.get("per_subtype",{}).get(s,{}).get("model_r2"),
                    "XGBoost MAE": ed.get("per_subtype",{}).get(s,{}).get("model_mae"),
                    "Random Forest R²": ed.get("per_subtype",{}).get(s,{}).get("rf_r2"),
                    "RF MAE": ed.get("per_subtype",{}).get(s,{}).get("rf_mae"),
                    "GNN R²": ed.get("per_subtype",{}).get(s,{}).get("gnn_r2"),
                    "GNN MAE": ed.get("per_subtype",{}).get(s,{}).get("gnn_mae"),
                } for s in SUBTYPES]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as e: st.error(f"GNN: {e}")
        else: st.info("Evaluation report not available.")

    with tabs[5]:
        ep = Path("outputs/external_validation/external_validation_report.json")
        if ep.exists():
            ed = json.loads(ep.read_text())
            c1, c2, c3 = st.columns(3)
            c1.metric("Novel Molecules", ed.get("n_novel_molecules",0))
            c2.metric("Successful", ed.get("n_successful_predictions",0))
            c3.metric("Errors", ed.get("n_errors",0))
            rows = []
            for sn, m in ed.get("per_subtype_metrics",{}).items():
                if sn == "selectivity_recall_at_1": continue
                rows.append({"Subtype": sn, "N": m.get("n"), "R²": "Insufficient" if m.get("insufficient_data") else f"{m.get('r2'):.3f}", "MAE": "N/A" if m.get("insufficient_data") else f"{m.get('mae'):.3f}"})
            if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True)
            sel = ed.get("per_subtype_metrics",{}).get("selectivity_recall_at_1",{})
            if sel: st.metric("Selectivity Recall@1", f"{sel.get('accuracy',0):.3f}", f"{sel.get('correct',0)}/{sel.get('total',0)}")
        else: st.info("External validation not available.")

    with tabs[6]:
        lp = Path("outputs/benchmark/benchmark_comparison.json")
        if lp.exists():
            ld = json.loads(lp.read_text())
            rows = [{"Model": k,
                "Method": v.get("method",""), "Split": v.get("split",""),
                **{f"{s} R²": v.get("metrics",{}).get(s,{}).get("r2") for s in SUBTYPES}
            } for k, v in ld.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else: st.info("Lit. benchmark not available.")

def _render_3d(mol_block):
    import json as _json
    esc = _json.dumps(mol_block)
    return f"""
    <!DOCTYPE html><html><head>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>body{{margin:0;padding:0;background:transparent}}#v{{width:100%;height:320px;border-radius:8px;border:1px solid #e2e6ed;background:#f8f9fa}}</style>
    </head><body><div id="v"></div>
    <script>$('#v').ready(function(){{ let v=$3Dmol.createViewer('v',{{backgroundColor:'#f8f9fa'}}); v.addModel({esc},'sdf'); v.setStyle({{}},{{stick:{{radius:0.2,colorscheme:'Jmol'}},sphere:{{radius:0.4,scale:0.3}}}}); v.zoomTo(); v.render(); }});
    </script></body></html>"""

def run_app():
    st.set_page_config(page_title="AR Selectivity Predictor", layout="wide")
    st.markdown(_MAIN_CSS, unsafe_allow_html=True)

    # ── Dynamic metrics from eval report ──
    rp = Path("outputs/validoutput/precise/evaluation_precise_report.json")
    default_metrics = {"overall_r2":"0.845","overall_mae":"0.396","n":"33,401",
        "A1":["0.809","0.403","8,272"], "A2A":["0.835","0.529","8,407"], "A2B":["0.801","0.305","8,290"], "A3":["0.894","0.347","8,432"]}
    m = default_metrics
    if rp.exists():
        try:
            ed = _load_json(str(rp)) or {}
            ov = ed.get("overall",{})
            if ov.get("model_r2") is not None:
                m["overall_r2"] = f"{ov['model_r2']:.3f}"
                m["overall_mae"] = f"{ov['model_mae']:.3f}"
                m["n"] = f"{ed.get('n_train',0)+ed.get('n_test',0):,}"
            for s in SUBTYPES:
                sd = ed.get("per_subtype",{}).get(s,{})
                if sd:
                    m[s] = [f"{sd.get('model_r2',0):.3f}", f"{sd.get('model_mae',0):.3f}", f"{sd.get('n_train',0)+sd.get('n_test',0):,}"]
        except Exception: pass

    # ── Hero Header ──
    st.markdown(f'''
    <div class="hero-card">
      <div class="hero-title">🧬 Adenosine Receptor Selectivity Predictor</div>
      <div class="hero-sub">
        Rapid <i>in silico</i> profiling of pChEMBL binding affinities across all four adenosine receptor subtypes
        (A<sub>1</sub>, A<sub>2A</sub>, A<sub>2B</sub>, A<sub>3</sub>) using a dual-architecture consensus strategy:
        XGBoost + Random Forest with conformal prediction bounds, and PyTorch Geometric GNN (GINE).
      </div>
      <div class="hero-badges">
        <span class="hero-badge">🎯 Overall R²: {m["overall_r2"]}</span>
        <span class="hero-badge">📉 MAE: {m["overall_mae"]}</span>
        <span class="hero-badge">🔬 {m["n"]} data points</span>
        <span class="hero-badge">🛡️ Conformal 90% CIs</span>
        <span class="hero-badge">🧠 XGBoost + RF + GNN</span>
        <span class="hero-badge">📐 Scaffold split CV</span>
      </div>
      <div class="metrics-dash">
        {"".join(f'<div class="metric-card"><div class="metric-label">A<sub>{s[1:].lower() if len(s)>1 else ""}</sub> Subtype</div><div class="metric-value">{m[s][0]}</div><div class="metric-sub">MAE: {m[s][1]} | n={m[s][2]}</div></div>' if isinstance(m.get(s),list) else "" for s in ["A1","A2A","A2B","A3"])}
      </div>
    </div>
    ''', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔬 Single Prediction", "📁 Batch Prediction", "📊 Results"])
    with tab1: _section_single_prediction()
    with tab2: _section_batch_prediction()
    with tab3: _section_results()

if __name__ == "__main__":
    run_app()
