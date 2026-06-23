from __future__ import annotations
import json, sys, pickle, numpy as np, time
from pathlib import Path
from datetime import datetime
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
:root{--nv:#0a1e3d;--nl:#1a3a6a;--te:#00b4d8;--gr:#2ecc71;--am:#f39c12;--re:#e74c3c;--bg:#f4f6f9;--ca:#fff;--bo:#e2e6ed;--tx:#1a2330;--t2:#5a6a7a}
.stApp{background:var(--bg);font-family:'Inter',sans-serif}
.hero{background:linear-gradient(135deg,#0a1e3d,#1a3a6a);border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1rem;color:#fff;position:relative;overflow:hidden}
.hero::after{content:'';position:absolute;top:-40%;right:-15%;width:350px;height:350px;background:radial-gradient(circle,rgba(0,180,216,0.1),transparent 70%);border-radius:50%}
.hero h1{font-size:1.3rem;font-weight:700;margin:0 0 .25rem;letter-spacing:-.02em;position:relative;z-index:1}
.hero p{font-size:.75rem;opacity:.8;max-width:600px;line-height:1.4;margin-bottom:.5rem;position:relative;z-index:1}
.hero .bg{display:flex;gap:.35rem;flex-wrap:wrap;position:relative;z-index:1}
.hero .bg span{display:inline-flex;align-items:center;gap:.2rem;background:rgba(255,255,255,.1);backdrop-filter:blur(4px);padding:.15rem .5rem;border-radius:14px;font-size:.65rem;border:1px solid rgba(255,255,255,.12)}
.hero .dash{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.4rem;margin-top:.5rem;position:relative;z-index:1}
.hero .dc{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:.35rem .6rem}
.hero .dc .dl{font-size:.55rem;text-transform:uppercase;letter-spacing:.04em;opacity:.7}
.hero .dc .dv{font-size:1rem;font-weight:700}
.hero .dc .ds{font-size:.6rem;opacity:.6}
.card{background:var(--ca);border:1px solid var(--bo);border-radius:10px;padding:.6rem .8rem;margin-bottom:.5rem}
.ct{font-size:.8rem;font-weight:600;margin-bottom:.3rem;display:flex;align-items:center;gap:.3rem}
.tag{display:inline-flex;align-items:center;gap:.15rem;padding:.08rem .35rem;border-radius:5px;font-size:.65rem;font-weight:500}
.tg{background:#e8f8f0;color:#1a7d4a}.ta{background:#fef3e2;color:#b0700a}.tr{background:#fde8e8;color:#b91c1c}.tb{background:#e0f0fe;color:#1a5a9c}
.mg{display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:.35rem}
.mi{background:var(--bg);border-radius:6px;padding:.25rem .4rem;text-align:center}
.mi .v{font-size:.85rem;font-weight:700}.mi .l{font-size:.55rem;color:var(--t2);text-transform:uppercase;letter-spacing:.02em}
.pb{height:5px;border-radius:3px;margin:.2rem 0;background:var(--bo);overflow:hidden}
.pb .f{height:100%;border-radius:3px;transition:width .4s ease}
.sd{height:1px;background:linear-gradient(90deg,transparent,var(--bo),transparent);margin:.6rem 0}
.stTextInput>div>div>input{border-radius:8px!important;border:2px solid var(--bo)!important;padding:.4rem .6rem!important;font-size:.8rem!important}
.stTextInput>div>div>input:focus{border-color:var(--te)!important;box-shadow:0 0 0 3px rgba(0,180,216,.1)!important}
.stButton>button{background:linear-gradient(135deg,#1a3a6a,#2a5a9a)!important;color:#fff!important;border:none!important;border-radius:8px!important;padding:.3rem 1rem!important;font-weight:500!important;font-size:.75rem!important;transition:all .15s!important}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 3px 10px rgba(26,58,106,.2)!important}
.stTabs [data-baseweb="tab-list"]{gap:0;border-radius:10px;background:var(--ca);padding:.2rem;border:1px solid var(--bo)}
.stTabs [data-baseweb="tab"]{border-radius:7px!important;padding:.25rem .7rem!important;font-size:.7rem!important;font-weight:500!important}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1a3a6a,#2a5a9a)!important;color:#fff!important}
.hi{display:flex;align-items:center;justify-content:space-between;padding:.2rem .4rem;border-radius:5px;font-size:.7rem;border:1px solid var(--bo);margin-bottom:.2rem}
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

    # Properties
    st.markdown('<div class="ct">📊 Properties</div><div class="mg">', unsafe_allow_html=True)
    for lb, vl in [("MW",d["MW"]),("LogP",d["LogP"]),("HBD",d["HBD"]),("HBA",d["HBA"]),
        ("RotB",d["RotBonds"]),("Arom",d["AromRings"]),("TPSA",d["TPSA"]),("QED",f"{qed:.2f}")]:
        st.markdown(f'<div class="mi"><div class="v">{vl}</div><div class="l">{lb}</div></div>', unsafe_allow_html=True)
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
                plt.title(f"{r['best_target']} SHAP",fontsize=9,fontweight="bold")
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
            fig,ax = plt.subplots(figsize=(5.5,2))
            for s in SUBTYPES:
                if s in rd.columns and rd[s].notna().any():
                    ax.hist(rd[s].dropna().astype(float), bins=20, alpha=.5, label=s)
            ax.axvline(6, color="#e74c3c", ls="--", lw=1)
            ax.set_xlabel("pChEMBL"); ax.legend(fontsize=7); plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.download_button("⬇ CSV", rd.to_csv(index=False).encode(), "ar_batch.csv", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Results ──
def _results():
    bd = "outputs/validoutput/precise"
    tabs = st.tabs(["📊 Metrics","🧩 SHAP/Y-Rand","📋 Diagnostics","🔍 Examples","🧠 GNN","🔬 External","📚 Lit."])
    with tabs[0]:
        try:
            o,p = load_evaluation_tables(bd)
            if not o.empty: st.markdown('<div class="card"><div class="ct">Overall</div>'); st.dataframe(o,use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
            if not p.empty: st.markdown('<div class="card"><div class="ct">Per Subtype</div>'); st.dataframe(p,use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
        except: pass
        for fp in ["outputs/validoutput/precise/calibration_precise_plot.png","outputs/validoutput/precise/calibration_root_plot.png","outputs/calibration_plot.png"]:
            if Path(fp).exists(): st.image(fp,use_container_width=True); break
    with tabs[1]:
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
                st.success("✅ True SAR" if not yd["leakage_warning"] else "⚠ Leakage!")
    with tabs[2]:
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
        try:
            st.dataframe(load_run_summary(bd),use_container_width=True)
            db,no=load_examples(bd)
            if not db.empty: st.markdown("**DB**"); st.dataframe(db,use_container_width=True)
            if not no.empty: st.markdown("**Novel**"); st.dataframe(no,use_container_width=True)
        except: pass
    with tabs[4]:
        ep=Path("outputs/validoutput/precise/evaluation_precise_report.json")
        if ep.exists():
            ed=_load_json(str(ep)) or {}
            rows=[{"Subtype":s,"XGB R²":ed.get("per_subtype",{}).get(s,{}).get("model_r2"),"XGB MAE":ed.get("per_subtype",{}).get(s,{}).get("model_mae"),"RF R²":ed.get("per_subtype",{}).get(s,{}).get("rf_r2"),"RF MAE":ed.get("per_subtype",{}).get(s,{}).get("rf_mae"),"GNN R²":ed.get("per_subtype",{}).get(s,{}).get("gnn_r2"),"GNN MAE":ed.get("per_subtype",{}).get(s,{}).get("gnn_mae")} for s in SUBTYPES]
            st.dataframe(pd.DataFrame(rows),use_container_width=True)
        else: st.info("N/A")
    with tabs[5]:
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
        lp=Path("outputs/benchmark/benchmark_comparison.json")
        if lp.exists():
            ld=json.loads(lp.read_text())
            rows=[{"Model":k,"Method":v.get("method",""),"Split":v.get("split",""),**{f"{s} R²":v.get("metrics",{}).get(s,{}).get("r2") for s in SUBTYPES}} for k,v in ld.items()]
            st.dataframe(pd.DataFrame(rows),use_container_width=True)
        else: st.info("N/A")

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
