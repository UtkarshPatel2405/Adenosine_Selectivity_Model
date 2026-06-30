# src/app/components/docking_panel.py
"""Docking panel — ligand (SMILES) vs best-predicted receptor, real API first."""
import streamlit as st
import pandas as pd
from src.docking import run_docking, ADENOSINE_TARGETS


def render_docking_panel(smiles: str, best_target: str):
    """Auto-selects receptor from best prediction, runs real docking API."""
    st.markdown('<div class="sd"></div>', unsafe_allow_html=True)
    target_info = ADENOSINE_TARGETS.get(best_target, ADENOSINE_TARGETS["A2A"])

    st.markdown(
        f'<div class="ct">🧬 Molecular Docking  <span class="tag tp">Ligand (SMILES) → {best_target} ({target_info["pdb"]})</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:.7rem;color:#94a3b8;line-height:1.5;margin-bottom:.5rem">'
        f'Ligand: <code style="font-size:.65rem;color:#e2e8f0">{smiles[:60]}{".." if len(smiles)>60 else ""}</code><br>'
        f'Receptor: <b>{target_info["name"]}</b> (PDB: <a href="https://www.rcsb.org/structure/{target_info["pdb"]}" target="_blank" style="color:#38bdf8">{target_info["pdb"]}</a>) — '
        f'auto-selected as highest predicted pChEMBL subtype.'
        f'</div>', unsafe_allow_html=True)

    dock_key = f"dock_{smiles[:20]}"
    if dock_key not in st.session_state:
        st.session_state[dock_key] = None

    if st.session_state[dock_key] is None:
        if st.button("🧬 Run Docking (API → Fallback)", use_container_width=True, key="dock_btn"):
            with st.status("Docking in progress…", expanded=True) as status:
                st.write("📡 Trying DockThor API…")
                result = run_docking(smiles, best_target)
                api = result.get("api_used", "local")
                if api == "DockThor":
                    status.update(label="✅ DockThor API — real docking completed", state="complete")
                elif api == "local":
                    status.update(label="⚠️ API unavailable — using local estimation", state="error")
                st.session_state[dock_key] = result
                st.rerun()
    else:
        result = st.session_state[dock_key]
        _display_results(result)
        if st.button("🔄 Re-run Docking", use_container_width=True, key="redock"):
            st.session_state[dock_key] = None
            st.rerun()


def _display_results(result: dict):
    """Table 1: Top 7 poses. Table 2: Molecular parameters."""
    method = result.get("method", "Unknown")
    api = result.get("api_used", "local")
    error = result.get("error")
    poses = result.get("poses", [])

    color = {"DockThor": "#22c55e", "SwissDock": "#f59e0b", "local": "#94a3b8"}.get(api, "#94a3b8")
    label = {"DockThor": "✅ Real Docking (DockThor Vina)", "SwissDock": "SwissDock", "local": "⚠️ Local Estimation — no API key"}.get(api, api)
    st.markdown(f'<span style="font-size:.7rem;background:rgba(34,197,94,.12);color:{color};padding:.15rem .4rem;border-radius:4px;border:1px solid {color}40">{label}</span>', unsafe_allow_html=True)
    if error:
        st.warning(error)

    if not poses:
        st.info("No docking poses returned. External API may be unavailable.")
        return

    # ── Table 1: Top 7 Poses ──
    st.markdown('<div class="ct" style="margin-top:.6rem">📊 Table 1: Top Docking Poses — Binding Scores & Affinity</div>', unsafe_allow_html=True)
    rows = []
    for p in poses:
        rows.append({
            "Rank": p.get("rank", "—"),
            "Target": p.get("subtype", "—"),
            "Score (kcal/mol)": p.get("score_kcal", p.get("score", "—")),
            "Est. Ki (µM)": p.get("ki_um", p.get("ki", "—")),
            "Lig. Efficiency": p.get("ligand_efficiency", "—"),
            "Receptor": p.get("receptor", "—"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True,
        column_config={
            "Score (kcal/mol)": st.column_config.NumberColumn(format="%.2f"),
            "Est. Ki (µM)": st.column_config.NumberColumn(format="%.2f"),
            "Lig. Efficiency": st.column_config.NumberColumn(format="%.3f"),
        })

    # ── Table 2: Molecular Parameters ──
    st.markdown('<div class="ct" style="margin-top:.6rem">⚗️ Table 2: Molecular Properties (from best pose)</div>', unsafe_allow_html=True)
    bp = poses[0] if poses else {}
    param_data = [
        ("LogP", bp.get("logp", "—")), ("MW (Da)", bp.get("mw", "—")),
        ("TPSA (Å²)", bp.get("tpsa", "—")), ("H-Bond Donors", bp.get("hbd", "—")),
        ("H-Bond Acceptors", bp.get("hba", "—")), ("Rotatable Bonds", bp.get("rot_bonds", "—")),
    ]
    cols = st.columns(3)
    for i, (label, val) in enumerate(param_data):
        with cols[i % 3]:
            st.markdown(
                f'<div class="card" style="text-align:center;padding:.3rem .5rem">'
                f'<div style="font-size:.55rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.03em">{label}</div>'
                f'<div style="font-size:1rem;font-weight:700;color:#f8fafc">{val}</div></div>',
                unsafe_allow_html=True)
