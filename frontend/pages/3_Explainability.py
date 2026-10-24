"""Explainability page: SHAP breakdown, plain-English rationale, what-if simulator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.graph_objects as go
import streamlit as st

from frontend.utils import api_client
from frontend.utils.forms import application_form
from frontend.utils.theme import apply_theme, page_header

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")
apply_theme()
page_header("Explainability", "SHAP-based rationale and what-if scenario simulator")

if not api_client.is_authenticated():
    st.warning("Please sign in on the home page first.")
    st.stop()

application = application_form("explain")

col_explain, col_cf = st.columns(2)
explain_clicked = col_explain.button("🔎 Explain this prediction", type="primary")
cf_clicked = col_cf.button("🧪 Run what-if simulator")

if explain_clicked:
    try:
        with st.spinner("Computing SHAP explanation..."):
            explanation = api_client.explain(application)
        st.session_state["explanation"] = explanation
    except api_client.ApiError as exc:
        st.error(str(exc))

if "explanation" in st.session_state:
    explanation = st.session_state["explanation"]
    st.subheader("Top contributing factors")
    contributions = sorted(explanation["contributions"], key=lambda c: abs(c["shap_value"]), reverse=True)[:12]
    features = [c["feature"] for c in contributions][::-1]
    values = [c["shap_value"] for c in contributions][::-1]
    colors = ["#f87171" if v > 0 else "#4ade80" for v in values]

    fig = go.Figure(go.Bar(x=values, y=features, orientation="h", marker_color=colors))
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(t=10),
        xaxis_title="SHAP contribution (impact on predicted default probability)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Plain-English explanation")
    st.info(explanation["plain_english"])

if cf_clicked:
    try:
        with st.spinner("Searching for a path to approval..."):
            cf = api_client.counterfactual(application)
        st.session_state["counterfactual"] = cf
    except api_client.ApiError as exc:
        st.error(str(exc))

if "counterfactual" in st.session_state:
    cf = st.session_state["counterfactual"]
    st.subheader("What-if scenario")
    c1, c2, c3 = st.columns(3)
    c1.metric("Original risk", f"{cf['original_probability']:.1%}")
    c2.metric("Simulated risk", f"{cf['final_probability']:.1%}")
    c3.metric("Outcome", "✅ Approval reachable" if cf["achieved"] else "⚠️ Not reachable in step budget")

    if cf["changes"]:
        st.write("Suggested changes, applied one at a time:")
        for i, change in enumerate(cf["changes"], start=1):
            st.markdown(
                f"**{i}.** `{change['feature']}`: {change['from_value']:.3g} → {change['to_value']:.3g} "
                f"&nbsp;→ risk drops to **{change['resulting_probability']:.1%}**",
                unsafe_allow_html=True,
            )
    else:
        st.write("No changes needed — this applicant already clears the decision threshold.")
