_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ═══════════════════════════════════════════════════════════
   Scientific Platform Theme - High Contrast & Accessibility
   ═══════════════════════════════════════════════════════════ */

html, body, [data-testid="stAppViewContainer"] {
    background: #090f1d !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

[data-testid="stHeader"] {
    background: rgba(9, 15, 29, 0.95) !important;
    backdrop-filter: blur(8px) !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.15) !important;
}

[data-testid="stSidebar"] {
    background: #060b15 !important;
    border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
}

/* ── Typography ─────────────────────────────────────────── */
h1, h2, h3 { color: #f8fafc !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
h1 { font-size: 1.6rem !important; }
h2 { font-size: 1.25rem !important; }
h3 { font-size: 1.0rem !important; }
p, span, label, div, li, td, th { color: #e2e8f0 !important; }
.stMarkdown p { color: #e2e8f0 !important; line-height: 1.6 !important; }
code { font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important; color: #f1f5f9 !important; }

/* ── Cards ────────────────────────────── */
.card {
    background: #111a2e !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 8px !important;
    padding: 0.8rem 1.0rem !important;
    margin-bottom: 0.8rem !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
}

/* ── Scientific Badges ──────────────────────────────────── */
.badge {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.25rem !important;
    padding: 0.18rem 0.5rem !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
    letter-spacing: 0.02em !important;
    white-space: nowrap !important;
}
.badge-blue { background: rgba(56, 189, 248, 0.2) !important; color: #38bdf8 !important; border: 1px solid rgba(56, 189, 248, 0.4) !important; }
.badge-green { background: rgba(34, 197, 94, 0.2) !important; color: #4ade80 !important; border: 1px solid rgba(34, 197, 94, 0.4) !important; }
.badge-amber { background: rgba(234, 179, 8, 0.2) !important; color: #facc15 !important; border: 1px solid rgba(234, 179, 8, 0.4) !important; }
.badge-red { background: rgba(239, 68, 68, 0.2) !important; color: #f87171 !important; border: 1px solid rgba(239, 68, 68, 0.4) !important; }
.badge-purple { background: rgba(168, 85, 247, 0.2) !important; color: #c084fc !important; border: 1px solid rgba(168, 85, 247, 0.4) !important; }
.badge-cyan { background: rgba(20, 184, 166, 0.2) !important; color: #2dd4bf !important; border: 1px solid rgba(20, 184, 166, 0.4) !important; }
.badge-slate { background: rgba(148, 163, 184, 0.2) !important; color: #cbd5e1 !important; border: 1px solid rgba(148, 163, 184, 0.35) !important; }

.badge-row {
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 0.35rem !important;
}

/* ── Section Headers ────────────────────────────────────── */
.section-header {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.6rem !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.15) !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

/* ── Hero Section ───────────────────────────────────────── */
.hero {
    padding: 1.2rem 1.6rem !important;
    background: #0f172a !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 10px !important;
    margin-bottom: 1.0rem !important;
}
.hero h1 { font-size: 1.5rem !important; font-weight: 800 !important; margin: 0 !important; color: #f8fafc !important; }
.hero p { font-size: 0.8rem !important; color: #cbd5e1 !important; margin: 0.3rem 0 0.6rem 0 !important; }

/* ── Dashboard Metrics ──────────────────────────────────── */
.dash-grid { display: flex !important; gap: 0.5rem !important; margin-top: 0.8rem !important; flex-wrap: wrap !important; }
.dash-card {
    background: #1e293b !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 6px !important;
    padding: 0.6rem 0.8rem !important;
    text-align: center !important;
    flex: 1 !important;
    min-width: 80px !important;
}
.dash-card:hover {
    border-color: rgba(56, 189, 248, 0.3) !important;
}
.dash-label { font-size: 0.6rem !important; font-weight: 600 !important; color: #cbd5e1 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; margin-bottom: 0.15rem !important; }
.dash-value { font-size: 1.2rem !important; font-weight: 700 !important; color: #ffffff !important; }
.dash-sub { font-size: 0.6rem !important; color: #94a3b8 !important; margin-top: 0.1rem !important; }

/* ── Progress bars ──────────────────────────────────────── */
.pb { background: rgba(148, 163, 184, 0.2) !important; border-radius: 4px !important; overflow: hidden !important; height: 5px !important; }
.pb .f { height: 100% !important; border-radius: 4px !important; }

/* ── Inputs & Selectboxes ────────────────────────────────── */
.stTextInput > div > div > input {
    background: #0f172a !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 6px !important;
    color: #f8fafc !important;
    padding: 0.5rem 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

div[data-baseweb="select"] input {
    caret-color: transparent !important;
}

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
    background: #0369a1 !important;
    color: #ffffff !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.4rem 0.8rem !important;
}
.stButton > button:hover {
    background: #0284c7 !important;
}

/* ── Tabs ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f172a !important;
    border-radius: 6px !important;
    padding: 0.15rem !important;
    border: 1px solid rgba(56, 189, 248, 0.08) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    padding: 0.3rem 0.6rem !important;
    font-weight: 500 !important;
    font-size: 0.75rem !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #f8fafc !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(56, 189, 248, 0.15) !important;
    color: #7dd3fc !important;
    font-weight: 600 !important;
}

/* ── Info / Tooltip Boxes ───────────────────────────────── */
.sci-box {
    background: rgba(56, 189, 248, 0.03) !important;
    border: 1px solid rgba(56, 189, 248, 0.1) !important;
    border-radius: 6px !important;
    padding: 0.75rem 0.9rem !important;
    font-size: 0.75rem !important;
    color: #b0bec5 !important;
    line-height: 1.5 !important;
    margin-bottom: 0.6rem !important;
}

/* ── Separator ──────────────────────────────────────────── */
.sd { height: 1px; background: rgba(56, 189, 248, 0.1) !important; margin: 0.8rem 0 !important; }

/* ── History items ──────────────────────────────────────── */
.hi {
    display: flex !important; align-items: center !important; gap: 0.3;
    padding: 0.35rem 0.5rem !important;
    background: #0f172a !important;
    border: 1px solid rgba(56, 189, 248, 0.05) !important;
    border-radius: 5px !important;
    margin-bottom: 0.3rem !important;
    font-size: 0.7rem !important;
}

/* ── DataFrames ─────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(56, 189, 248, 0.1) !important;
    border-radius: 6px !important;
}

/* ── Expanders ──────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #0f172a !important;
    border: 1px solid rgba(56, 189, 248, 0.08) !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}
.streamlit-expanderContent {
    background: #0b1120 !important;
    border: 1px solid rgba(56, 189, 248, 0.08) !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
    padding: 0.75rem 0.9rem !important;
}

/* ── Subtype affinity pills ─────────────────────────────── */
.affinity-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    padding: 0.3rem 0.5rem !important;
    margin-bottom: 0.3rem !important;
    background: #111a2e !important;
    border: 1px solid rgba(56, 189, 248, 0.05) !important;
    border-radius: 5px !important;
}

/* ── Feature importance table ───────────────────────────── */
.feat-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    padding: 0.18rem 0.35rem !important;
    font-size: 0.72rem !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.05) !important;
}

/* ── Metric containers ──────────────────────────────────── */
[data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #b0bec5 !important; font-size: 0.72rem !important; }

</style>
"""
