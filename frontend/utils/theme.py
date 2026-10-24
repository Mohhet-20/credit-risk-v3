"""
Dark-theme styling and shared UI chrome for the Streamlit frontend.

Design decision: rather than relying solely on Streamlit's built-in theme
config (`.streamlit/config.toml`, which we also ship for the base palette),
a small CSS injection adds the "premium fintech dashboard" details that
config.toml can't express — card shadows, gradient accents, KPI tiles, and
subtle hover transitions — while staying inside a single stylesheet block so
it's trivial to audit or rip out.
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
    .stApp { background-color: #0e1117; }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2b 0%, #151922 100%);
        border: 1px solid #2a3040;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.45);
    }
    div[data-testid="stMetricLabel"] { color: #9aa4b8 !important; font-weight: 500; }
    div[data-testid="stMetricValue"] { color: #f0f2f6 !important; }

    .risk-badge {
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem; letter-spacing: 0.02em;
    }
    .risk-very_low  { background: #113d2b; color: #4ade80; }
    .risk-low       { background: #123a2e; color: #34d399; }
    .risk-medium    { background: #3d3311; color: #facc15; }
    .risk-high      { background: #3d2311; color: #fb923c; }
    .risk-very_high { background: #3d1414; color: #f87171; }

    .decision-approved { color: #4ade80; font-weight: 700; }
    .decision-rejected { color: #f87171; font-weight: 700; }
    .decision-manual_review { color: #facc15; font-weight: 700; }

    section[data-testid="stSidebar"] { background-color: #12151c; border-right: 1px solid #22262f; }

    h1, h2, h3 { letter-spacing: -0.01em; }

    .platform-header {
        display: flex; align-items: center; gap: 12px; margin-bottom: 0.25rem;
    }
    .platform-badge {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white; padding: 2px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;
    }

    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="platform-header"><h1 style="margin:0">{title}</h1>'
        f'<span class="platform-badge">v2.0</span></div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)


def risk_badge_html(risk_band: str) -> str:
    label = risk_band.replace("_", " ").title()
    return f'<span class="risk-badge risk-{risk_band}">{label}</span>'


def decision_html(decision: str) -> str:
    label = decision.replace("_", " ").title()
    return f'<span class="decision-{decision}">{label}</span>'
