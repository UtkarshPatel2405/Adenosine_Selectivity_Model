# src/app/pages/model_results.py
"""Model results page — all 7 tabs with fallback messages when no data."""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
from src.config import SUBTYPES
from src.app.components.model_reports import load_evaluation_tables, load_run_summary, load_examples, _load_json

BD = "outputs/validoutput/precise"

def _methodology():
    st.markdown('''
    <div class="section-header">📐 Pipeline Methodology</div>
    <div class="card" style="padding:.8rem 1rem;margin-bottom:.6rem">
    <div style="font-size:.7rem;color:#94a3b8;line-height:1.5;margin-bottom:.6rem">
    <b>Pipeline Overview.</b> 33,401 pChEMBL values across 4 adenosine receptor subtypes curated from ChEMBL.
    Morgan FP (2048), MACCS (167), 15 RDKit descriptors = 2,230 features.
    Scaffold split (Murcko, 80/20) ensures OOD generalization.
    </div>''', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card card-glow anim-in" style="text-align:center;padding:.5rem .3rem"><div class="section-header" style="font-size:.55rem;justify-content:center;margin-bottom:.2rem">📚 Data Curation</div><div style="color:#94a3b8;font-size:.55rem">ChEMBL + Lit · 33K values · 4 subtypes</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card card-glow anim-in-d1" style="text-align:center;padding:.5rem .3rem"><div class="section-header" style="font-size:.55rem;justify-content:center;margin-bottom:.2rem">🧪 Featurization</div><div style="color:#94a3b8;font-size:.55rem">Morgan (2048) · MACCS (167) · RDKit (15)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card card-glow anim-in-d2" style="text-align:center;padding:.5rem .3rem"><div class="section-header" style="font-size:.55rem;justify-content:center;margin-bottom:.2rem">🔀 Scaffold Split</div><div style="color:#94a3b8;font-size:.55rem">Murcko 80/20 · Novel chemotypes</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="display:flex;flex-wrap:wrap;gap:.4rem;justify-content:center;margin:.5rem 0">'
        '<span class="badge badge-blue">🌲 XGBoost</span>'
        '<span class="badge badge-green">🌳 RandomForest</span>'
        '<span class="badge badge-amber">🧠 GNN (MPNN)</span>'
        '<span class="badge badge-purple">🛡️ Conformal Prediction (MAPIE CV+ 90% CI)</span>'
        '<span class="badge badge-cyan">🎯 Prediction</span>'
        '</div>', unsafe_allow_html=True)

    st.markdown('''
    <div style="display:flex;flex-wrap:wrap;gap:.4rem;justify-content:center;margin:.5rem 0 .3rem;font-size:.6rem;color:#64748b">
    <span>↓ </span> <span class="badge badge-blue">XGBoost</span>
    <span>→ </span> <span class="badge badge-green">RandomForest</span>
    <span>→ </span> <span class="badge badge-amber">GNN</span>
    <span>→ </span> <span class="badge badge-purple">Conformal</span>
    <span>→ </span> <span class="badge badge-cyan">Ensemble</span>
    </div>''', unsafe_allow_html=True)

    st.markdown('<div style="font-size:.55rem;color:#64748b;margin-top:.3rem"><b>Conformal Prediction:</b> Each prediction is accompanied by a 90% confidence interval. Coverage (the fraction of true values within predicted intervals) is the primary reliability metric — target: ≥0.85.</div>', unsafe_allow_html=True)


def _no_data(tab_name="this tab"):
    st.markdown(
        f'<div style="text-align:center;padding:1.5rem;color:#64748b;font-size:.75rem">'
        f'No {tab_name} data available. Run <code>python src/retrain_production.py</code> to generate evaluation reports.'
        f'</div>', unsafe_allow_html=True)


def render_model_results():
    has_data = Path(f"{BD}/evaluation_precise_report.json").exists()
    if not has_data:
        st.markdown(
            '<div class="card" style="text-align:center;padding:.8rem;margin-bottom:.5rem">'
            '<span style="font-size:.75rem;color:#94a3b8">📊 Model evaluation reports not yet generated.</span>'
            '<span style="font-size:.65rem;color:#64748b;margin-left:.3rem">Run <code>python src/retrain_production.py</code></span>'
            '</div>', unsafe_allow_html=True)

    tabs = st.tabs(["📊 Metrics", "🧩 SHAP/Y-Rand", "📋 Diagnostics", "🔍 Examples", "🧠 GNN", "🔬 External", "📐 Method."])

    with tabs[0]:
        st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)
        st.markdown('<div class="sci-box"><b>Performance.</b> Scaffold-split OOD test (20%). Conformal coverage = fraction of 90% intervals containing truth. MAE reported in pChEMBL units (typical assay noise: ±0.3–0.5 pChEMBL).</div>', unsafe_allow_html=True)
        if not has_data:
            _no_data("metrics")
        else:
            try:
                o, p = load_evaluation_tables(BD)
                if not o.empty:
                    st.markdown('<b style="font-size:.7rem;color:#e2e8f0">Overall Metrics</b>', unsafe_allow_html=True)
                    st.dataframe(o, use_container_width=True, hide_index=True)
                if not p.empty:
                    st.markdown('<b style="font-size:.7rem;color:#e2e8f0">Per Subtype Metrics</b>', unsafe_allow_html=True)
                    st.dataframe(p, use_container_width=True, hide_index=True)
            except Exception as e: st.caption(f"Tables: {e}")
            c1, c2 = st.columns(2)
            with c1:
                for fp in ["outputs/validoutput/precise/calibration_precise_plot.png", "outputs/calibration_plot.png"]:
                    if Path(fp).exists(): st.image(fp, use_container_width=True, caption="Calibration Plot"); break
                st.markdown('<div style="font-size:.55rem;color:#64748b">Calibration curve: predicted vs observed coverage for 90% conformal prediction intervals.</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div style="font-size:.55rem;color:#64748b;margin-bottom:.3rem">MAE by uncertainty quartile — higher-uncertainty bins should have higher error.</div>', unsafe_allow_html=True)
                rp = Path(f"{BD}/evaluation_precise_report.json")
                if rp.exists():
                    ed = _load_json(str(rp)) or {}
                    cq = ed.get("overall", {}).get("calibration_quartiles")
                    if cq:
                        fig = go.Figure(go.Bar(x=[f"Q{q['bin']}" for q in cq], y=[q["mae_mean"] for q in cq],
                            marker_color=["#00b4d8","#2ecc71","#f39c12","#e74c3c"],
                            text=[f"{q['mae_mean']:.3f}" for q in cq], textposition="outside"))
                        fig.update_layout(height=220, margin=dict(t=20,b=10,l=10,r=10),
                            xaxis_title="Uncertainty Quartile", yaxis_title="MAE",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#94a3b8", size=10))
                        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.markdown('<div class="section-header">🧩 SHAP & Y-Randomization</div>', unsafe_allow_html=True)
        st.markdown('<div class="sci-box"><b>SHAP</b> bar + beeswarm plots (left) show global feature importance. <b>Y-Randomization</b> (right) tests whether the model captures true SAR: real R² must exceed the shuffled distribution by >3σ to rule out chance correlation.</div>', unsafe_allow_html=True)
        if not has_data:
            _no_data("SHAP/Y-Randomization")
        else:
            sc = st.selectbox("Subtype", SUBTYPES, key="sy")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<b style="font-size:.65rem;color:#e2e8f0">SHAP Feature Importance</b>', unsafe_allow_html=True)
                for f in [Path(f"outputs/shap/{sc}_bar.png"), Path(f"outputs/shap/{sc}_beeswarm.png")]:
                    if f.exists(): st.image(str(f), use_container_width=True)
            with c2:
                st.markdown('<b style="font-size:.65rem;color:#e2e8f0">Y-Randomization Test</b>', unsafe_allow_html=True)
                for p in [Path(f"outputs/y_randomization/{sc}_distribution.png")]:
                    if p.exists(): st.image(str(p), use_container_width=True)
                yr = Path(f"outputs/y_randomization/{sc}_report.json")
                if yr.exists():
                    yd = json.loads(yr.read_text())
                    c1m, c2m, c3m = st.columns(3)
                    c1m.metric("Real R²", f"{yd['real_r2']:.3f}")
                    c2m.metric("Shuffled μ", f"{yd['shuffled_r2_mean']:.3f}")
                    c3m.metric("Shuffled σ", f"{yd['shuffled_r2_std']:.3f}")
                    if not yd.get("leakage_warning", True):
                        st.markdown('<span class="badge badge-green">✅ True SAR — model captures genuine structure-activity relationships</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge badge-amber">⚠ Potential leakage — R² indistinguishable from shuffled</span>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="section-header">📋 Diagnostics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sci-box"><b>Diagnostics.</b> Scaffold diversity, activity cliffs (Tanimoto≥0.85, ΔpChEMBL≥1.0), pChEMBL distributions — assess dataset quality and coverage.</div>', unsafe_allow_html=True)
        if not has_data:
            _no_data("diagnostics")
        else:
            dg = st.selectbox("Target", ["Combined","A1","A2A","A2B","A3"], key="dg")
            if dg == "Combined":
                dp = Path("outputs/diagnostics/combined_diagnosis_report.json")
                if dp.exists():
                    dd = json.loads(dp.read_text())
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Compounds", dd["n_compounds"])
                    c2.metric("Scaffolds", dd["scaffold_diversity"]["n_unique_scaffolds"])
                    c3.metric("Diversity", f"{dd['scaffold_diversity']['diversity_ratio']:.3f}")
                if Path("outputs/diagnostics/combined_pchembl_distribution.png").exists():
                    st.image("outputs/diagnostics/combined_pchembl_distribution.png", use_container_width=True)
            else:
                pf = dg.lower(); dp = Path(f"outputs/diagnostics/{pf}_diagnosis_report.json")
                if dp.exists():
                    dd = json.loads(dp.read_text())
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Compounds", dd["n_compounds"])
                        st.metric("Scaffolds", dd["scaffold_diversity"]["n_unique_scaffolds"])
                    with c2:
                        st.metric("Cliffs", dd.get("n_activity_cliffs", 0))
                        st.metric("pChEMBL", f'{dd["pchembl_stats"]["mean"]:.2f}±{dd["pchembl_stats"]["std"]:.2f}')
                    for px in ["pchembl_distribution", "activity_cliffs_shifts"]:
                        f = Path(f"outputs/diagnostics/{pf}_{px}.png")
                        if f.exists(): st.image(str(f), use_container_width=True)

    with tabs[3]:
        st.markdown('<div class="section-header">🔍 Example Predictions</div>', unsafe_allow_html=True)
        st.markdown('<div class="sci-box"><b>Example Predictions.</b> Database-held + novel molecules across all subtypes — compare predicted vs experimental pChEMBL for selected test-set molecules.</div>', unsafe_allow_html=True)
        if not has_data:
            _no_data("example predictions")
        else:
            try:
                sm = load_run_summary(BD); db, no = load_examples(BD)
                if not sm.empty: st.dataframe(sm, use_container_width=True, hide_index=True)
                if not db.empty:
                    st.markdown('<b style="font-size:.65rem;color:#e2e8f0">Database molecules</b>', unsafe_allow_html=True)
                    st.dataframe(db, use_container_width=True, hide_index=True)
                if not no.empty:
                    st.markdown('<b style="font-size:.65rem;color:#e2e8f0">Novel molecules</b>', unsafe_allow_html=True)
                    st.dataframe(no, use_container_width=True, hide_index=True)
            except: pass

    with tabs[4]:
        st.markdown('<div class="section-header">🧠 Model Comparison (XGBoost vs RF vs GNN)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sci-box"><b>XGBoost vs RF vs GNN.</b> Bar charts compare R² and MAE across subtypes. The scatter plot shows XGBoost vs GNN R² — points above the diagonal favor GNN, below favor XGBoost.</div>', unsafe_allow_html=True)
        if not has_data:
            _no_data("GNN comparison")
        else:
            ep = Path(f"{BD}/evaluation_precise_report.json")
            if ep.exists():
                ed = _load_json(str(ep)) or {}
                gp = Path("outputs/gnn/all_subtypes_summary.json")
                gd = json.loads(gp.read_text())["results"] if gp.exists() else {}
                rows = []
                for s in SUBTYPES:
                    sd = ed.get("per_subtype", {}).get(s, {}); g = gd.get(s, {})
                    rows.append({"Subtype": s, "XGB R²": sd.get("model_r2"), "XGB MAE": sd.get("model_mae"),
                        "RF R²": sd.get("rf_r2"), "RF MAE": sd.get("rf_mae"), "GNN R²": g.get("r2"), "GNN MAE": g.get("mae")})
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                c1, c2 = st.columns(2)
                with c1:
                    fig = go.Figure()
                    for m, co, k in [("XGBoost","#00b4d8","XGB R²"),("RF","#2ecc71","RF R²"),("GNN","#f39c12","GNN R²")]:
                        y_vals = pd.to_numeric(df[k], errors='coerce').astype(float)
                        fig.add_trace(go.Bar(name=m, x=df["Subtype"], y=y_vals, marker_color=co, text=y_vals.round(3), textposition="outside"))
                    fig.update_layout(barmode="group", height=260, margin=dict(t=10,b=10,l=10,r=10), yaxis_title="R²",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=10),
                          legend=dict(orientation="h", y=1.05, x=0))
                    fig.update_yaxes(gridcolor="rgba(255,255,255,.05)")
                    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = go.Figure()
                    for m, co, k in [("XGBoost","#00b4d8","XGB MAE"),("RF","#2ecc71","RF MAE"),("GNN","#f39c12","GNN MAE")]:
                        y_vals = pd.to_numeric(df[k], errors='coerce').astype(float)
                        fig.add_trace(go.Bar(name=m, x=df["Subtype"], y=y_vals, marker_color=co, text=y_vals.round(3), textposition="outside"))
                    fig.update_layout(barmode="group", height=260, margin=dict(t=10,b=10,l=10,r=10), yaxis_title="MAE",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=10),
                          legend=dict(orientation="h", y=1.05, x=0))
                    fig.update_yaxes(gridcolor="rgba(255,255,255,.05)")
                    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)")
                    st.plotly_chart(fig, use_container_width=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["XGB R²"], y=df["GNN R²"], mode="markers+text", text=df["Subtype"],
                    textposition="top center", marker=dict(size=12, color="#00b4d8", line=dict(color="#fff", width=1))))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="rgba(255,255,255,.2)"), showlegend=False))
                fig.update_layout(height=260, margin=dict(t=10,b=10,l=10,r=10), xaxis_title="XGBoost R²", yaxis_title="GNN R²",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=10))
                fig.update_xaxes(range=[0,1], gridcolor="rgba(255,255,255,.05)")
                fig.update_yaxes(range=[0,1], gridcolor="rgba(255,255,255,.05)")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.markdown('<div class="section-header">🔬 External Validation</div>', unsafe_allow_html=True)
        st.markdown('<div class="sci-box"><b>External Validation.</b> Novel molecules from literature testing generalization beyond scaffold split — a stricter test of real-world applicability.</div>', unsafe_allow_html=True)
        ep = Path("outputs/external_validation/external_validation_report.json")
        if not ep.exists():
            _no_data("external validation")
        else:
            ed = json.loads(ep.read_text())
            c1, c2, c3 = st.columns(3)
            c1.metric("Molecules", ed.get("n_novel_molecules", 0))
            c2.metric("OK", ed.get("n_successful_predictions", 0))
            c3.metric("Errors", ed.get("n_errors", 0))
            rows = []
            for sn, m in ed.get("per_subtype_metrics", {}).items():
                if sn == "selectivity_recall_at_1": continue
                rows.append({"Subtype": sn, "N": m.get("n"), "R²": "—" if m.get("insufficient_data") else f'{m.get("r2"):.3f}', "MAE": "—" if m.get("insufficient_data") else f'{m.get("mae"):.3f}'})
            if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tabs[6]:
        _methodology()
        lp = Path("outputs/benchmark/benchmark_comparison.json")
        if lp.exists():
            st.markdown('<div class="sd"></div>', unsafe_allow_html=True)
            ld = json.loads(lp.read_text())
            rows = [{"Model": k, "Method": v.get("method",""), "Split": v.get("split",""),
                **{f"{s} R²": v.get("metrics",{}).get(s,{}).get("r2") for s in SUBTYPES}} for k,v in ld.items()]
            st.markdown('<div class="section-header">📚 Literature Benchmarks</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
