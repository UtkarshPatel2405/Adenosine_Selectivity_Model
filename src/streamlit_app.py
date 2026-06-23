from __future__ import annotations
import json, sys, pickle, numpy as np, time
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
from src.app.components.model_reports import load_evaluation_tables, load_run_summary, load_examples, _load_json

for k in ("history","history_df","pred"):
    if k not in st.session_state:
        st.session_state[k] = [] if k == "history" else pd.DataFrame() if k == "history_df" else None

# ── Caching ──
@st.cache_data(show_spinner=False)
def _cached_predict(smiles: str, threshold: float) -> dict:
    return predict(smiles, threshold)

@st.cache_resource(show_spinner=False)
def _load_shap_model(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

# ── CSS ──
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --nv:#0a1e3d; --nl:#1a3a6a; --te:#00b4d8; --gr:#2ecc71; --am:#f39c12; --re:#e74c3c;
  --bg:linear-gradient(135deg,#0f1729 0%,#1a2744 50%,#0f1729 100%);
  --ca:#1e2d50; --ca2:#253660; --bo:#2d4068; --tx:#e8edf5; --t2:#8899bb;
  --sh:0 4px 20px rgba(0,0,0,.25); --sh-sm:0 2px 8px rgba(0,0,0,.18);
  --gl:radial-gradient(ellipse at 20% 50%, rgba(0,180,216,0.06) 0%, transparent 60%);
}
.stApp{background:var(--bg);font-family:'Inter',sans-serif;color:var(--tx)}
.hero{background:linear-gradient(135deg,#0d1b3a,#1a3a6a);border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1rem;color:#fff;position:relative;overflow:hidden;box-shadow:var(--sh)}
.hero::after{content:'';position:absolute;top:-40%;right:-15%;width:350px;height:350px;background:radial-gradient(circle,rgba(0,180,216,0.12),transparent 70%);border-radius:50%}
.hero h1{font-size:1.3rem;font-weight:700;margin:0 0 .25rem;letter-spacing:-.02em;position:relative;z-index:1}
.hero p{font-size:.75rem;opacity:.8;max-width:600px;line-height:1.4;margin-bottom:.5rem;position:relative;z-index:1}
.hero .bg{display:flex;gap:.35rem;flex-wrap:wrap;position:relative;z-index:1}
.hero .bg span{display:inline-flex;align-items:center;gap:.2rem;background:rgba(255,255,255,.08);backdrop-filter:blur(4px);padding:.15rem .5rem;border-radius:14px;font-size:.65rem;border:1px solid rgba(255,255,255,.1)}
.hero .dash{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.4rem;margin-top:.5rem;position:relative;z-index:1}
.hero .dc{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:.35rem .6rem}
.hero .dc .dl{font-size:.55rem;text-transform:uppercase;letter-spacing:.04em;opacity:.7}
.hero .dc .dv{font-size:1rem;font-weight:700}
.hero .dc .ds{font-size:.6rem;opacity:.6}
.card{background:var(--ca);border:1px solid var(--bo);border-radius:10px;padding:.6rem .8rem;margin-bottom:.5rem;box-shadow:var(--sh-sm)}
.card-glow{background:linear-gradient(135deg,var(--ca),var(--ca2));border:1px solid var(--bo);border-radius:10px;padding:.6rem .8rem;margin-bottom:.5rem;box-shadow:var(--sh-sm)}
.ct{font-size:.8rem;font-weight:600;margin-bottom:.3rem;display:flex;align-items:center;gap:.3rem;color:var(--tx)}
.tag{display:inline-flex;align-items:center;gap:.15rem;padding:.08rem .35rem;border-radius:5px;font-size:.65rem;font-weight:500}
.tg{background:rgba(46,204,113,.15);color:#5ce69a}.ta{background:rgba(243,156,18,.15);color:#f5c542}.tr{background:rgba(231,76,60,.15);color:#f0807a}.tb{background:rgba(0,180,216,.15);color:#56d4f0}
.mg{display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:.35rem}
.mi{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);border-radius:8px;padding:.35rem .5rem;text-align:center}
.mi .v{font-size:.85rem;font-weight:700;color:var(--tx)}.mi .l{font-size:.55rem;color:var(--t2);text-transform:uppercase;letter-spacing:.02em}
.pb{height:5px;border-radius:3px;margin:.2rem 0;background:rgba(255,255,255,.08);overflow:hidden}
.pb .f{height:100%;border-radius:3px;transition:width .4s ease}
.sd{height:1px;background:linear-gradient(90deg,transparent,var(--bo),transparent);margin:.6rem 0}
.stTextInput>div>div>input{border-radius:8px!important;border:2px solid var(--bo)!important;padding:.4rem .6rem!important;font-size:.8rem!important;background:var(--ca)!important;color:var(--tx)!important}
.stTextInput>div>div>input:focus{border-color:var(--te)!important;box-shadow:0 0 0 3px rgba(0,180,216,.15)!important}
.stTextInput>div>div>input::placeholder{color:var(--t2)!important;opacity:.6}
.stButton>button{background:linear-gradient(135deg,#1a3a6a,#2a5a9a)!important;color:#fff!important;border:none!important;border-radius:8px!important;padding:.3rem 1rem!important;font-weight:500!important;font-size:.75rem!important;transition:all .15s!important}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 3px 14px rgba(26,58,106,.3)!important}
.stTabs [data-baseweb="tab-list"]{gap:0;border-radius:10px;background:var(--ca);padding:.2rem;border:1px solid var(--bo)}
.stTabs [data-baseweb="tab"]{border-radius:7px!important;padding:.25rem .7rem!important;font-size:.7rem!important;font-weight:500!important;color:var(--t2)!important}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1a3a6a,#2a5a9a)!important;color:#fff!important}
.hi{display:flex;align-items:center;justify-content:space-between;padding:.2rem .4rem;border-radius:5px;font-size:.7rem;border:1px solid var(--bo);margin-bottom:.2rem;background:rgba(255,255,255,.03)}
.st-bb,.st-at,.st-cb,.st-da,.st-cv,.st-dj,.st-ds{background-color:transparent!important}
.stSelectbox>div>div{background:var(--ca)!important;border:1px solid var(--bo)!important;color:var(--tx)!important;border-radius:8px!important}
.stSelectbox>div>div>div{color:var(--tx)!important}
.stDataFrame{background:var(--ca)!important;color:var(--tx)!important}
.stDataFrame table{color:var(--tx)!important}
.st-dc,.st-db{color:var(--t2)!important}
.stDownloadButton>button{background:rgba(0,180,216,.1)!important;border:1px solid var(--te)!important;color:var(--te)!important;border-radius:6px!important;font-size:.7rem!important}
.stExpander>div>div>div>div{color:var(--tx)!important}
.stExpander>div{border:1px solid var(--bo)!important;border-radius:8px!important;background:var(--ca)!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b3a,#15203f)!important;border-right:1px solid var(--bo)!important}
[data-testid="stSidebar"] *{color:var(--tx)!important}
.stMetric label{color:var(--t2)!important}
.stMetric [data-testid="stMetricValue"]{color:var(--tx)!important}
</style>"""

def _tag(v, t=6.0):
    p = float(v) if v is not None else 0
    return ("tg","Active") if p>=t else ("ta","Weak") if p>=t-1.5 else ("tr","Inact")

def _imp(v):
    return ("tb","N/A") if v is None else ("tg","High") if v>=0.6 else ("ta","Med") if v>=0.4 else ("tr","Low")

# ── Sidebar ──
def _sidebar():
    with st.sidebar:
        st.markdown("### 🧬 Session")
        st.markdown(f"**{len(st.session_state.history)}** molecules scanned")
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.history = []; st.session_state.history_df = pd.DataFrame(); st.rerun()
        st.markdown("---")
        if st.session_state.history:
            for h in reversed(st.session_state.history[-8:]):
                sm = h["smiles"][:25]+".." if len(h["smiles"])>25 else h["smiles"]
                c,_ = _tag(h["bv"])
                st.markdown(f'<div class="hi"><span>{sm}</span><span class="tag {c}">{h["bt"]}</span><span style="font-size:.55rem;color:var(--t2)">{h["t"]}</span></div>', unsafe_allow_html=True)
            if not st.session_state.history_df.empty:
                csv = st.session_state.history_df.to_csv(index=False).encode()
                st.download_button("⬇ Export CSV", csv, "ar_session.csv", use_container_width=True)

# ── Single Prediction ──
def _single():
    EXAMPLES = {
        "Adenosine":"C1=NC2=C(C(=N1)N)N=CN2[C@@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O",
        "Caffeine":"CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Theophylline":"CN1C2=C(C(=O)N(C1=O)C)NC=N2",
        "Regadenoson":"C1=NC2=C(C(=N1)N)N=CN2[C@@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O",
        "Custom":"CCn1c(/N=C/c2ccc(Br)cc2)c(C#N)sc1=S",
    }
    ci, cx = st.columns([3,1])
    with ci:
        smiles = st.text_input("SMILES", value="CCn1c(/N=C/c2ccc(Br)cc2)c(C#N)sc1=S", label_visibility="collapsed",
            placeholder="Enter SMILES, press Enter, then click Predict…")
    with cx:
        eg = st.selectbox("", list(EXAMPLES.keys()), label_visibility="collapsed")
        if eg and eg != "Custom": smiles = EXAMPLES[eg]

    if st.button("🔬 Predict", use_container_width=True):
        with st.spinner("Running 3 models + conformal intervals…"):
            t0 = time.time()
            mb, mn, mx = generate_3d_conformer(smiles)
            sv = draw_2d_svg(smiles)
            try: r = _cached_predict(smiles, 6.0)
            except Exception as e: st.error(f"Error: {e}"); return
            el = time.time() - t0
            r["_elapsed"] = el; r["_mol_block"] = mb; r["_svg"] = sv; r["_smiles"] = smiles
            r["_profile"] = qed_profile(smiles)
            st.session_state.pred = r
            xgb_best = r["predictions"]["XGBoost"].get(r["best_target"],0)
            st.session_state.history.append({"smiles":r["smiles"],"bt":r["best_target"],"bv":xgb_best,
                "t":datetime.now().strftime("%H:%M"),"name":r["smiles"][:20]})
            row = {"SMILES":r["smiles"],"Best":r["best_target"]}
            for s in SUBTYPES: row[s]=r["predictions"]["XGBoost"].get(s)
            row["Source"]=r["source"]
            st.session_state.history_df = pd.concat([st.session_state.history_df, pd.DataFrame([row])], ignore_index=True)

    r = st.session_state.pred
    if r is None:
        st.info("Enter a SMILES and click **Predict** to begin.")
        return

    canon = r["smiles"]; preds = r["predictions"]; unc = r["uncertainty"]
    d = r["descriptors"]; prof = r.get("_profile",{}) or {}; qed = prof.get("QED",0) if isinstance(prof,dict) else 0
    mb = r.get("_mol_block"); sv = r.get("_svg")
    el = r.get("_elapsed",0)
    st.markdown(f'<div class="sd"></div>', unsafe_allow_html=True)

    # Structure
    s1, s2 = st.columns(2)
    with s1:
        if sv:
            st.markdown('<div class="card"><div class="ct">🧪 2D</div>', unsafe_allow_html=True)
            st.image(sv, use_container_width=True)
            st.download_button("⬇ SVG", sv, "struct.svg", mime="image/svg+xml", key="sd1")
            st.markdown('</div>', unsafe_allow_html=True)
    with s2:
        if mb:
            st.markdown('<div class="card"><div class="ct">🧬 3D</div>', unsafe_allow_html=True)
            st.components.v1.html(_render_3d(mb), height=280)
            ca, cb = st.columns(2)
            with ca: st.download_button("⬇ SDF", mb, "conf.sdf", key="sd2")
            with cb:
                pdb = generate_pdb_block(smiles)
                if pdb: st.download_button("⬇ PDB", pdb, "conf.pdb", key="sd3")
            st.markdown('</div>', unsafe_allow_html=True)

    src = "Database hit" if r["in_database"] else "ML Ensemble"
    st.markdown(f'<span class="tag tb">{src}</span> <span style="font-size:.65rem;color:var(--t2)">{el:.1f}s</span>', unsafe_allow_html=True)

    # Properties - visual card with horizontal bars
    _prop_config = [
        ("MW", d["MW"], 50, 600, "Da", "#00b4d8"),
        ("LogP", d["LogP"], -2, 6, "", "#2ecc71"),
        ("HBD", d["HBD"], 0, 10, "donors", "#f39c12"),
        ("HBA", d["HBA"], 0, 14, "acceptors", "#e74c3c"),
        ("RotB", d["RotBonds"], 0, 15, "bonds", "#9b59b6"),
        ("Arom", d["AromRings"], 0, 8, "rings", "#1abc9c"),
        ("TPSA", d["TPSA"], 0, 200, "Å²", "#3498db"),
        ("QED", qed, 0, 1, "", "#e67e22"),
    ]
    st.markdown('<div class="card-glow"><div class="ct">📊 Molecular Properties</div>', unsafe_allow_html=True)
    for lb, vl, vmin, vmax, unit, color in _prop_config:
        pct = min(max((float(vl) - vmin) / (vmax - vmin) * 100, 0), 100)
        vl_str = f"{vl:.1f}" if isinstance(vl, float) and vl != int(vl) else str(vl)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.5rem;margin:.15rem 0">'
            f'<span style="font-size:.65rem;font-weight:500;min-width:2.5rem;color:var(--t2)">{lb}</span>'
            f'<div class="pb" style="flex:1;height:6px"><div class="f" style="width:{pct}%;background:linear-gradient(90deg,{color},rgba(255,255,255,.3))"></div></div>'
            f'<span style="font-size:.75rem;font-weight:600;min-width:3.5rem;text-align:right">{vl_str} <span style="font-weight:400;color:var(--t2)">{unit}</span></span>'
            f'</div>', unsafe_allow_html=True)
    st.markdown('</div>')

    # Affinity
    st.markdown(f'<div class="sd"></div><div class="ct">🎯 Affinity  <span class="tag tg">Best: {r["best_target"]}</span></div>', unsafe_allow_html=True)
    sm = st.selectbox("Model", ["XGBoost","RandomForest","PyTorch"], label_visibility="collapsed")
    for k in SUBTYPES:
        v = preds.get(sm,{}).get(k,0); c, l = _tag(v)
        pct = min(float(v)/10*100,100); co = "#2ecc71" if v>=6 else "#f39c12" if v>=4.5 else "#e74c3c"
        uv = unc.get(sm,{}).get(k,0)
        st.markdown(
            f'<div class="card" style="padding:.35rem .7rem">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;font-size:.8rem">'
            f'<b>A<sub>{k[1:].lower()}</sub></b>'
            f'<span><span class="tag {c}">{l}</span> <b style="font-size:.9rem;margin-left:.3rem">{v:.2f}</b></span></div>'
            f'<div class="pb"><div class="f" style="width:{pct}%;background:{co}"></div></div>'
            f'<div style="font-size:.6rem;color:var(--t2);display:flex;justify-content:space-between">'
            f'<span>{sm}</span><span>σ={uv:.3f}</span></div></div>', unsafe_allow_html=True)

    # SHAP
    if r["source"] == "model":
        with st.expander("🔬 SHAP: Why this prediction?"):
            st.markdown(
                '<div class="card" style="font-size:.7rem;color:var(--t2);line-height:1.5">'
                "<b>How to interpret SHAP values:</b> "
                "SHAP (SHapley Additive exPlanations) attributes each feature's contribution to the predicted pChEMBL score. "
                "Features pushing the prediction <span style='color:#e74c3c'>right</span> (red) increase activity; "
                "features pushing <span style='color:#3498db'>left</span> (blue) decrease it. "
                "The <b>waterfall</b> plot below shows the top-5 contributing features for this molecule, "
                "starting from the baseline (mean prediction across the training set). "
                "Use this to understand which molecular substructures drive selectivity for your compound."
                '</div>', unsafe_allow_html=True)
            try:
                from src.features import build_features
                mp = Path(f"models/precise/xgboost_{r['best_target']}_production.pkl")
                if not mp.exists(): mp = Path(f"models/precise/xgboost_precise_{r['best_target'].lower()}_model.pkl")
                if not mp.exists(): mp = Path(f"models/xgboost_{r['best_target'].lower()}_model.pkl")
                mc = _load_shap_model(mp)
                et = mc if type(mc).__name__ not in ("CrossConformalRegressor","MapieRegressor") else (
                    mc._mapie_regressor.estimator_.estimators_[0] if type(mc).__name__=="CrossConformalRegressor" else mc.estimators_[0])
                sp = Path("models/precise/scaler_precise.pkl")
                if not sp.exists(): sp = Path("models/scaler.pkl")
                with open(sp,"rb") as f: pl = pickle.load(f)
                fn = [f"FP{i}" for i in range(2048)]+[f"MAC{i}" for i in range(167)]+list(pl.feature_filter.feature_names)
                x = build_features(canon,pl).reshape(1,-1)
                xdf = pd.DataFrame(x,columns=fn)
                e = shap.TreeExplainer(et); sv = e(xdf)
                fig,ax = plt.subplots(figsize=(6,2.8))
                shap.plots.waterfall(sv[0],max_display=5,show=False)
                plt.title(f"{r['best_target']} SHAP Waterfall",fontsize=9,fontweight="bold")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            except Exception as ex: st.caption(f"SHAP: {ex}")

    # Safety row
    st.markdown('<div class="sd"></div>', unsafe_allow_html=True)
    ca, cb, cc = st.columns(3)
    with ca:
        al = check_pains(smiles)
        st.markdown(f'<div class="card"><div class="ct">{("⚠️","✅")[not al]} Safety</div><span class="tag {("tr","tg")[not al]}">{f"{len(al)} PAINS" if al else "Clean"}</span></div>', unsafe_allow_html=True)
    with cb:
        sim = nearest_tanimoto(smiles)
        if sim is not None:
            ic, il = _imp(sim)
            st.markdown(f'<div class="card"><div class="ct">🎯 AD</div><span class="tag {ic}">{il} ({sim:.3f})</span></div>', unsafe_allow_html=True)
    with cc:
        st.markdown(f'<div class="card"><div class="ct">💊 Drug-like</div><span class="tag tb">QED {qed:.3f}</span></div>', unsafe_allow_html=True)

    # Top-5 similar
    with st.expander("🔗 Top-5 similar training molecules"):
        try:
            _, ts = topk_tanimoto(smiles, k=5)
            if ts:
                st.dataframe(pd.DataFrame([{"#":i+1,"SMILES":s,"Tanimoto":f"{t:.4f}"} for i,(s,t) in enumerate(ts)]).set_index("#"),
                    use_container_width=True, hide_index=True)
        except Exception as ex: st.caption(f"Similarity: {ex}")

# ── Batch ──
def _batch():
    st.markdown('<div class="card"><div class="ct">📁 Batch CSV Prediction</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    if up is None:
        st.info("Upload CSV → get per-subtype predictions + hit analysis"); st.markdown('</div>', unsafe_allow_html=True); return
    df = pd.read_csv(up); sc = _infer_smiles_col(df)
    st.markdown(f'**{sc}** · {len(df)} rows')
    if st.button("🚀 Run Batch", use_container_width=True):
        with st.spinner(f"Predicting {len(df)} molecules…"):
            rd = predict_batch(df, 6.0, smiles_col=sc, mode="precise")
            errs = rd["error"].notna().sum() if "error" in rd.columns else 0
            ok = len(rd)-errs
            st.markdown('<div class="mg">', unsafe_allow_html=True)
            for lb,vl in [("Total",len(rd)),("OK",ok),("Errors",errs),("Hit%",f"{(rd['best_target'].notna()&rd['best_target'].isin(SUBTYPES)).sum()/ok*100:.0f}%" if ok else "0")]:
                st.markdown(f'<div class="mi"><div class="v" style="color:#1a5a9c">{vl}</div><div class="l">{lb}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>')
            dc = [c for c in [sc,'A1','A2A','A2B','A3','best_target'] if c in rd.columns]
            st.dataframe(rd[dc], use_container_width=True, height=250)
            fig = go.Figure()
            colors = {"A1":"#00b4d8","A2A":"#2ecc71","A2B":"#f39c12","A3":"#e74c3c"}
            for s in SUBTYPES:
                if s in rd.columns and rd[s].notna().any():
                    fig.add_trace(go.Histogram(x=rd[s].dropna().astype(float), name=s, opacity=.6,
                        marker_color=colors.get(s,"#3498db"), nbinsx=20))
            fig.add_vline(x=6, line_dash="dash", line_color="#e74c3c", line_width=1.5,
                annotation_text="Active (6.0)", annotation_position="top left")
            fig.update_layout(barmode="overlay", height=250, margin=dict(t=10,b=10,l=10,r=10),
                xaxis_title="pChEMBL", yaxis_title="Count",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8899bb",size=10),
                legend=dict(orientation="h", y=1.05, x=0),
                hovermode="x unified")
            fig.update_xaxes(gridcolor="rgba(255,255,255,.05)")
            fig.update_yaxes(gridcolor="rgba(255,255,255,.05)")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇ CSV", rd.to_csv(index=False).encode(), "ar_batch.csv", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Results ──
def _results():
    bd = "outputs/validoutput/precise"
    tabs = st.tabs(["📊 Metrics","🧩 SHAP/Y-Rand","📋 Diagnostics","🔍 Examples","🧠 GNN","🔬 External","📐 Method."])
    with tabs[0]:
        st.markdown(
            '<div class="card" style="font-size:.7rem;color:var(--t2);line-height:1.5">'
            "<b>Model Performance Summary.</b> "
            "Metrics reported on a held-out scaffold-split test set (20%). "
            "R² quantifies variance explained; MAE gives the average absolute prediction error in pChEMBL units. "
            "Conformal coverage shows the fraction of test points where the 90% prediction interval contains the true value. "
            "Values closer to 90% indicate well-calibrated uncertainty estimates."
            '</div>', unsafe_allow_html=True)
        try:
            o,p = load_evaluation_tables(bd)
            if not o.empty:
                st.markdown('<div class="card"><div class="ct">Overall</div>', unsafe_allow_html=True)
                st.dataframe(o,use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            if not p.empty:
                st.markdown('<div class="card"><div class="ct">Per Subtype</div>', unsafe_allow_html=True)
                st.dataframe(p,use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        except: pass
        ci, c2 = st.columns([1,1])
        with ci:
            for fp in ["outputs/validoutput/precise/calibration_precise_plot.png","outputs/validoutput/precise/calibration_root_plot.png","outputs/calibration_plot.png"]:
                if Path(fp).exists():
                    st.markdown('<div class="card"><div class="ct">Calibration Plot</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div style="font-size:.65rem;color:var(--t2);line-height:1.4;margin-bottom:.4rem">'
                        "Binned calibration: test points grouped by predicted uncertainty into 4 quartiles. "
                        "Well-calibrated models show MAE increasing monotonically with uncertainty bin."
                        '</div>', unsafe_allow_html=True)
                    st.image(fp,use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    break
        with c2:
            rp = Path("outputs/validoutput/precise/evaluation_precise_report.json")
            if rp.exists():
                ed = _load_json(str(rp)) or {}
                dt = ed.get("overall",{})
                if dt.get("calibration_quartiles"):
                    cq = dt["calibration_quartiles"]
                    qfig = go.Figure()
                    qfig.add_trace(go.Bar(x=[f"Q{q['bin']}" for q in cq], y=[q["mae_mean"] for q in cq],
                        marker_color=["#00b4d8","#2ecc71","#f39c12","#e74c3c"],
                        text=[f"{q['mae_mean']:.3f}" for q in cq], textposition="outside"))
                    qfig.update_layout(height=220, margin=dict(t=20,b=10,l=10,r=10),
                        xaxis_title="Uncertainty Quartile", yaxis_title="Mean MAE",
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#8899bb",size=10),
                        title=dict(text="Calibration by Quartile", font=dict(size=12, color="#e8edf5")))
                    qfig.update_xaxes(gridcolor="rgba(255,255,255,.05)")
                    qfig.update_yaxes(gridcolor="rgba(255,255,255,.05)")
                    st.markdown('<div class="card"><div class="ct">MAE by Uncertainty Bin</div>', unsafe_allow_html=True)
                    st.plotly_chart(qfig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            # Residuals
            for fp in ["outputs/residuals_xgboost.png","outputs/pred_xgboost.png"]:
                if Path(fp).exists():
                    st.image(fp, use_container_width=True)
    with tabs[1]:
        st.markdown(
            '<div class="card" style="font-size:.7rem;color:var(--t2);line-height:1.5">'
            "<b>SHAP Global Feature Importance &amp; Y-Randomization Validation.</b> "
            "<b>Bar plot</b> (left): mean absolute SHAP value per feature across all test molecules. "
            "Higher bars indicate features with greater impact on predictions. "
            "<b>Beeswarm</b> (left, below): each dot is a molecule; color shows feature value (red=high, blue=low). "
            "Wide spread along the x-axis means the feature has a large influence. "
            "<b>Y-Randomization distribution</b> (right): R² scores from 20 models trained on shuffled activity values. "
            "If the real model R² falls within this distribution, the model may be detecting random correlations (data leakage). "
            "A real R² far above the shuffled distribution (+3σ) confirms true structure–activity relationships (SAR)."
            '</div>', unsafe_allow_html=True)
        sc = st.selectbox("Subtype", SUBTYPES, key="sy")
        c1,c2 = st.columns(2)
        with c1:
            for f in [Path(f"outputs/shap/{sc}_bar.png"),Path(f"outputs/shap/{sc}_beeswarm.png")]:
                if f.exists(): st.image(str(f),use_container_width=True)
        with c2:
            for p in [Path(f"outputs/y_randomization/{sc}_distribution.png")]:
                if p.exists(): st.image(str(p),use_container_width=True)
            yr = Path(f"outputs/y_randomization/{sc}_report.json")
            if yr.exists():
                yd=json.loads(yr.read_text()); st.metric("Real R²",f"{yd['real_r2']:.3f}")
                st.metric("Shuffled",f"{yd['shuffled_r2_mean']:.3f}±{yd['shuffled_r2_std']:.3f}")
                if not yd.get("leakage_warning", True):
                    st.success("✅ True SAR (real R² > 3σ above shuffled)")
                else:
                    st.warning("⚠ Potential leakage (real R² within shuffled distribution)")
    with tabs[2]:
        st.markdown(
            '<div class="card" style="font-size:.7rem;color:var(--t2);line-height:1.5">'
            "<b>Dataset Diagnostics.</b> "
            "<b>Scaffold diversity</b> measures how many unique Murcko scaffolds exist relative to the total compound count — "
            "lower ratios indicate a structurally homogeneous dataset. "
            "<b>Activity cliffs</b> identify pairs of structurally similar molecules (Tanimoto ≥ 0.85) with large pChEMBL differences (≥ 1.0) — "
            "these highlight where small structural changes dramatically affect activity. "
            "<b>pChEMBL distribution</b> shows the range and balance of activity values across the dataset."
            '</div>', unsafe_allow_html=True)
        dg = st.selectbox("Target",["Combined","A1","A2A","A2B","A3"],key="dg")
        if dg=="Combined":
            dp=Path("outputs/diagnostics/combined_diagnosis_report.json")
            if dp.exists():
                dd=json.loads(dp.read_text()); c1,c2,c3=st.columns(3)
                c1.metric("Compounds",dd["n_compounds"]); c2.metric("Scaffolds",dd["scaffold_diversity"]["n_unique_scaffolds"]); c3.metric("Diversity",f"{dd['scaffold_diversity']['diversity_ratio']:.3f}")
                if Path("outputs/diagnostics/combined_pchembl_distribution.png").exists(): st.image("outputs/diagnostics/combined_pchembl_distribution.png",use_container_width=True)
        else:
            pf=dg.lower(); dp=Path(f"outputs/diagnostics/{pf}_diagnosis_report.json")
            if dp.exists():
                dd=json.loads(dp.read_text()); c1,c2=st.columns(2)
                with c1: st.metric("Compounds",dd["n_compounds"]); st.metric("Scaffolds",dd["scaffold_diversity"]["n_unique_scaffolds"])
                with c2: st.metric("Cliffs",dd.get("n_activity_cliffs",0)); st.metric("pChEMBL",f'{dd["pchembl_stats"]["mean"]:.2f}±{dd["pchembl_stats"]["std"]:.2f}')
                for px in ["pchembl_distribution","activity_cliffs_shifts"]:
                    f=Path(f"outputs/diagnostics/{pf}_{px}.png")
                    if f.exists(): st.image(str(f),use_container_width=True)
    with tabs[3]:
        st.markdown(
            '<div class="card" style="font-size:.7rem;color:var(--t2);line-height:1.5">'
            "<b>Example Predictions.</b> "
            "Below are representative predictions for both database-held and novel molecules across all adenosine subtypes. "
            "The 'Hits' column lists which receptors the molecule is predicted to be active against (pChEMBL ≥ 6.0). "
            "'Similarity (AD)' shows the applicability domain score — high similarity to training data means higher confidence."
            '</div>', unsafe_allow_html=True)
        try:
            sm, db, no = load_run_summary(bd), *load_examples(bd)
            if not sm.empty: st.dataframe(sm,use_container_width=True)
            if not db.empty:
                st.markdown("**Database molecules** (experimental values known)")
                st.dataframe(db,use_container_width=True)
            if not no.empty:
                st.markdown("**Novel molecules** (ML-predicted only)")
                st.dataframe(no,use_container_width=True)
        except: pass
    with tabs[4]:
        st.markdown(
            '<div class="card" style="font-size:.7rem;color:var(--t2);line-height:1.5">'
            "<b>Model Comparison: XGBoost vs Random Forest vs GNN.</b> "
            "The GNN (MPNN/GINE) directly learns from molecular graph structure, while XGBoost and RF use "
            "pre-computed Morgan fingerprints + physicochemical descriptors. "
            "Bar charts show R² and MAE per subtype side-by-side. "
            "Scatter points compare XGBoost vs GNN R² across subtypes — points above the diagonal indicate XGBoost superiority."
            '</div>', unsafe_allow_html=True)
        ep=Path("outputs/validoutput/precise/evaluation_precise_report.json")
        if ep.exists():
            ed=_load_json(str(ep)) or {}
            gnn_summary = Path("outputs/gnn/all_subtypes_summary.json")
            gnn_data = json.loads(gnn_summary.read_text())["results"] if gnn_summary.exists() else {}
            rows=[]
            for s in SUBTYPES:
                sd=ed.get("per_subtype",{}).get(s,{})
                g = gnn_data.get(s,{})
                rows.append({"Subtype":s,"XGB R²":sd.get("model_r2"),"XGB MAE":sd.get("model_mae"),
                    "RF R²":sd.get("rf_r2"),"RF MAE":sd.get("rf_mae"),
                    "GNN R²":g.get("r2"),"GNN MAE":g.get("mae")})
            df = pd.DataFrame(rows)
            st.dataframe(df,use_container_width=True)
            # Bar charts
            c1, c2 = st.columns(2)
            with c1:
                r2fig = go.Figure()
                for model, color, key in [("XGBoost","#00b4d8","XGB R²"),("RF","#2ecc71","RF R²"),("GNN","#f39c12","GNN R²")]:
                    r2fig.add_trace(go.Bar(name=model, x=df["Subtype"], y=df[key],
                        marker_color=color, text=df[key].round(3), textposition="outside"))
                r2fig.update_layout(barmode="group", height=260, margin=dict(t=10,b=10,l=10,r=10),
                    yaxis_title="R²", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8899bb",size=10),
                    title=dict(text="R² by Model", font=dict(size=11, color="#e8edf5")),
                    legend=dict(orientation="h", y=1.05, x=0))
                r2fig.update_xaxes(gridcolor="rgba(255,255,255,.05)")
                r2fig.update_yaxes(gridcolor="rgba(255,255,255,.05)", range=[0,1])
                st.plotly_chart(r2fig, use_container_width=True)
            with c2:
                maefig = go.Figure()
                for model, color, key in [("XGBoost","#00b4d8","XGB MAE"),("RF","#2ecc71","RF MAE"),("GNN","#f39c12","GNN MAE")]:
                    maefig.add_trace(go.Bar(name=model, x=df["Subtype"], y=df[key],
                        marker_color=color, text=df[key].round(3), textposition="outside"))
                maefig.update_layout(barmode="group", height=260, margin=dict(t=10,b=10,l=10,r=10),
                    yaxis_title="MAE", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8899bb",size=10),
                    title=dict(text="MAE by Model", font=dict(size=11, color="#e8edf5")),
                    legend=dict(orientation="h", y=1.05, x=0))
                maefig.update_xaxes(gridcolor="rgba(255,255,255,.05)")
                maefig.update_yaxes(gridcolor="rgba(255,255,255,.05)")
                st.plotly_chart(maefig, use_container_width=True)
            # Scatter: XGB vs GNN R²
            scfig = go.Figure()
            scfig.add_trace(go.Scatter(x=df["XGB R²"], y=df["GNN R²"], mode="markers+text",
                text=df["Subtype"], textposition="top center",
                marker=dict(size=12, color="#00b4d8", line=dict(color="#fff", width=1))))
            scfig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                line=dict(dash="dash", color="rgba(255,255,255,.2)"), showlegend=False))
            scfig.update_layout(height=260, margin=dict(t=10,b=10,l=10,r=10),
                xaxis_title="XGBoost R²", yaxis_title="GNN R²",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8899bb",size=10),
                title=dict(text="XGBoost vs GNN R²", font=dict(size=11, color="#e8edf5")))
            scfig.update_xaxes(gridcolor="rgba(255,255,255,.05)", range=[0,1])
            scfig.update_yaxes(gridcolor="rgba(255,255,255,.05)", range=[0,1])
            st.plotly_chart(scfig, use_container_width=True)
        else: st.info("N/A")
    with tabs[5]:
        st.markdown(
            '<div class="card" style="font-size:.7rem;color:var(--t2);line-height:1.5">'
            "<b>External Validation.</b> "
            "Novel molecules (not in the training set) sourced from recent literature and public databases. "
            "These molecules test generalization to truly unseen chemical space beyond the scaffold split."
            '</div>', unsafe_allow_html=True)
        ep=Path("outputs/external_validation/external_validation_report.json")
        if ep.exists():
            ed=json.loads(ep.read_text()); c1,c2,c3=st.columns(3)
            c1.metric("Molecules",ed.get("n_novel_molecules",0)); c2.metric("OK",ed.get("n_successful_predictions",0)); c3.metric("Errors",ed.get("n_errors",0))
            rows=[]
            for sn,m in ed.get("per_subtype_metrics",{}).items():
                if sn=="selectivity_recall_at_1": continue
                rows.append({"Subtype":sn,"N":m.get("n"),"R²":"—" if m.get("insufficient_data") else f'{m.get("r2"):.3f}',"MAE":"—" if m.get("insufficient_data") else f'{m.get("mae"):.3f}'})
            if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True)
        else: st.info("N/A")
    with tabs[6]:
        st.markdown(
            '<div class="card" style="font-size:.7rem;color:var(--t2);line-height:1.5">'
            "<b>Model Methodology.</b> "
            "The pipeline integrates three complementary modeling approaches within a conformal prediction framework. "
            "Each approach captures different aspects of molecular structure–activity relationships."
            '</div>', unsafe_allow_html=True)
        _render_methodology_diagram()

def _render_3d(mb):
    import json as _j; e=_j.dumps(mb)
    return f'''<!DOCTYPE html><html><head>
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>body{{margin:0;padding:0;background:transparent}}#v{{width:100%;height:270px;border-radius:8px;border:1px solid #e2e6ed;background:#f8f9fa}}</style>
</head><body><div id="v"></div><script>
$('#v').ready(function(){{let v=$3Dmol.createViewer('v',{{backgroundColor:'#f8f9fa'}});v.addModel({e},'sdf');
v.setStyle({{}},{{stick:{{radius:.2,colorscheme:'Jmol'}},sphere:{{radius:.4,scale:.3}}}});v.zoomTo();v.render();}});
</script></body></html>'''

def _render_methodology_diagram():
    """SVG-style methodology flowchart rendered via HTML + CSS."""
    st.markdown(f'''
    <div class="card" style="padding:.8rem 1rem">
    <div style="display:flex;flex-wrap:wrap;gap:.5rem;justify-content:center;font-size:.7rem">
      <div style="background:linear-gradient(135deg,#0d1b3a,#1a3a6a);border-radius:8px;padding:.4rem .7rem;text-align:center;min-width:100px;border:1px solid rgba(0,180,216,.2)">
        <div style="font-size:1.2rem;margin-bottom:.2rem">📚</div>
        <div style="font-weight:600;font-size:.65rem">Data Curation</div>
        <div style="color:var(--t2);font-size:.55rem">ChEMBL +文献<br/>33K bioactivities</div>
      </div>
      <div style="display:flex;align-items:center;color:var(--t2);font-size:1rem">→</div>
      <div style="background:linear-gradient(135deg,#0d1b3a,#1a3a6a);border-radius:8px;padding:.4rem .7rem;text-align:center;min-width:100px;border:1px solid rgba(46,204,113,.2)">
        <div style="font-size:1.2rem;margin-bottom:.2rem">🧪</div>
        <div style="font-weight:600;font-size:.65rem">Featurization</div>
        <div style="color:var(--t2);font-size:.55rem">Morgan FP (2048)<br/>MACCS (167)<br/>RDKit desc.</div>
      </div>
      <div style="display:flex;align-items:center;color:var(--t2);font-size:1rem">→</div>
      <div style="background:linear-gradient(135deg,#0d1b3a,#1a3a6a);border-radius:8px;padding:.4rem .7rem;text-align:center;min-width:100px;border:1px solid rgba(243,156,18,.2)">
        <div style="font-size:1.2rem;margin-bottom:.2rem">🔀</div>
        <div style="font-weight:600;font-size:.65rem">Scaffold Split</div>
        <div style="color:var(--t2);font-size:.55rem">Murcko scaffolds<br/>80/20 train/test</div>
      </div>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:.5rem;justify-content:center;margin-top:.5rem;font-size:.7rem">
      <div style="background:linear-gradient(135deg,#0d1b3a,#1a3a6a);border-radius:8px;padding:.4rem .7rem;text-align:center;min-width:90px;border:1px solid rgba(0,180,216,.3)">
        <div style="font-size:1rem;margin-bottom:.2rem">🌲</div>
        <div style="font-weight:600;font-size:.65rem">XGBoost</div>
        <div style="color:var(--t2);font-size:.55rem">Tree ensemble<br/>CV-optimized</div>
      </div>
      <div style="background:linear-gradient(135deg,#0d1b3a,#1a3a6a);border-radius:8px;padding:.4rem .7rem;text-align:center;min-width:90px;border:1px solid rgba(46,204,113,.3)">
        <div style="font-size:1rem;margin-bottom:.2rem">🌳</div>
        <div style="font-weight:600;font-size:.65rem">RandomForest</div>
        <div style="color:var(--t2);font-size:.55rem">300 trees<br/>Bagged ensemble</div>
      </div>
      <div style="background:linear-gradient(135deg,#0d1b3a,#1a3a6a);border-radius:8px;padding:.4rem .7rem;text-align:center;min-width:90px;border:1px solid rgba(243,156,18,.3)">
        <div style="font-size:1rem;margin-bottom:.2rem">🧠</div>
        <div style="font-weight:600;font-size:.65rem">GNN (MPNN)</div>
        <div style="color:var(--t2);font-size:.55rem">Graph conv.<br/>GINE architecture</div>
      </div>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:.5rem;justify-content:center;margin-top:.5rem;font-size:.7rem">
      <div style="display:flex;align-items:center;color:var(--t2);font-size:1rem">↓</div>
      <div style="background:linear-gradient(135deg,#2a1a3d,#4a2a6a);border-radius:8px;padding:.4rem .7rem;text-align:center;min-width:120px;border:1px solid rgba(155,89,182,.3)">
        <div style="font-size:1rem;margin-bottom:.2rem">🛡️</div>
        <div style="font-weight:600;font-size:.65rem">Conformal Prediction</div>
        <div style="color:var(--t2);font-size:.55rem">MAPIE CV+<br/>90% prediction intervals</div>
      </div>
      <div style="display:flex;align-items:center;color:var(--t2);font-size:1rem">↓</div>
      <div style="background:linear-gradient(135deg,#1a3d2a,#2a6a4a);border-radius:8px;padding:.6rem .9rem;text-align:center;min-width:100px;border:1px solid rgba(46,204,113,.4);box-shadow:0 0 12px rgba(46,204,113,.15)">
        <div style="font-size:1.2rem;margin-bottom:.2rem">🎯</div>
        <div style="font-weight:600;font-size:.7rem;color:#5ce69a">Prediction</div>
        <div style="color:var(--t2);font-size:.55rem">pChEMBL ± σ<br/>A1/A2A/A2B/A3</div>
      </div>
    </div>
    </div>
    ''', unsafe_allow_html=True)

def run_app():
    st.set_page_config(page_title="AR Selectivity Predictor", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    rp = Path("outputs/validoutput/precise/evaluation_precise_report.json")
    dm = {"r2":"0.845","mae":"0.396","n":"33,401","A1":["0.809","0.403","8,272"],"A2A":["0.835","0.529","8,407"],"A2B":["0.801","0.305","8,290"],"A3":["0.894","0.347","8,432"]}
    m = dm
    if rp.exists():
        try:
            ed = _load_json(str(rp)) or {}; ov = ed.get("overall",{})
            if ov.get("model_r2") is not None:
                m["r2"]=f"{ov['model_r2']:.3f}"; m["mae"]=f"{ov['model_mae']:.3f}"; m["n"]=f"{ed.get('n_train',0)+ed.get('n_test',0):,}"
            for s in SUBTYPES:
                sd=ed.get("per_subtype",{}).get(s,{})
                if sd: m[s]=[f"{sd.get('model_r2',0):.3f}",f"{sd.get('model_mae',0):.3f}",f"{sd.get('n_train',0)+sd.get('n_test',0):,}"]
        except: pass
    st.markdown(f'''
    <div class="hero">
      <h1>🧬 Adenosine Receptor Selectivity Predictor</h1>
      <p>Rapid <i>in silico</i> pChEMBL profiling across A<sub>1</sub>, A<sub>2A</sub>, A<sub>2B</sub>, A<sub>3</sub>.
      XGBoost + RF + conformal prediction + PyTorch GNN.</p>
      <div class="bg">
        <span>🎯 R² {m["r2"]}</span><span>📉 MAE {m["mae"]}</span><span>🔬 {m["n"]}</span>
        <span>🛡️ 90% CIs</span><span>🧠 XGB+RF+GNN</span><span>📐 Scaffold CV</span>
      </div>
      <div class="dash">
        {"".join(f'<div class="dc"><div class="dl">A<sub>{s[1:].lower()}</sub></div><div class="dv">{m[s][0]}</div><div class="ds">MAE {m[s][1]}&nbsp;|&nbsp;n={m[s][2]}</div></div>' if isinstance(m.get(s),list) else "" for s in ["A1","A2A","A2B","A3"])}
      </div>
    </div>''', unsafe_allow_html=True)
    _sidebar()
    t1,t2,t3 = st.tabs(["🔬 Single","📁 Batch","📊 Results"])
    with t1: _single()
    with t2: _batch()
    with t3: _results()

if __name__ == "__main__":
    run_app()
