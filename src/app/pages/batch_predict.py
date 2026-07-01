# src/app/pages/batch_predict.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.config import SUBTYPES
from src.app.components.batch_predict import predict_batch, _infer_smiles_col

def render_batch_predict():
    st.markdown('<div class="card"><div class="ct">📁 Batch CSV Prediction</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    if up is None:
        st.info("Upload CSV → get per-subtype predictions + hit analysis")
        st.markdown('</div>', unsafe_allow_html=True)
        return
        
    try:
        df = pd.read_csv(up, encoding='utf-8')
    except UnicodeDecodeError:
        up.seek(0)
        df = pd.read_csv(up, encoding='latin1')
    sc = _infer_smiles_col(df)
    st.markdown(f'**{sc}** · {len(df)} rows')
    
    if st.button("🚀 Run Batch", use_container_width=True):
        with st.spinner(f"Predicting {len(df)} molecules…"):
            rd = predict_batch(df, 6.0, smiles_col=sc, mode="precise")
            errs = rd["error"].notna().sum() if "error" in rd.columns else 0
            ok = len(rd) - errs
            cols = st.columns(4)
            vals = [len(rd), ok, errs, f"{rd['best_target'].notna().sum() / ok * 100:.0f}%" if ok else "0"]
            for col, lb, vl in zip(cols, ["Total","OK","Errors","Hit%"], vals):
                col.metric(lb, vl)
            dc = [c for c in [sc, 'A1', 'A2A', 'A2B', 'A3', 'best_target'] if c in rd.columns]
            st.dataframe(rd[dc], use_container_width=True, height=250)
            
            fig = go.Figure()
            colors = {"A1": "#00b4d8", "A2A": "#2ecc71", "A2B": "#f39c12", "A3": "#e74c3c"}
            for s in SUBTYPES:
                if s in rd.columns and rd[s].notna().any():
                    fig.add_trace(go.Histogram(x=rd[s].dropna().astype(float), name=s, opacity=.6,
                        marker_color=colors.get(s, "#3498db"), nbinsx=20))
            fig.add_vline(x=6, line_dash="dash", line_color="#e74c3c", line_width=1.5,
                annotation_text="Active (6.0)", annotation_position="top left")
            fig.update_layout(barmode="overlay", height=250, margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="pChEMBL", yaxis_title="Count",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8899bb", size=10),
                legend=dict(orientation="h", y=1.05, x=0),
                hovermode="x unified")
            fig.update_xaxes(gridcolor="rgba(255,255,255,.05)")
            fig.update_yaxes(gridcolor="rgba(255,255,255,.05)")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇ CSV", rd.to_csv(index=False).encode(), "ar_batch.csv", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
