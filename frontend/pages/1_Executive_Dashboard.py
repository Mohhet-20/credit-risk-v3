"""Executive dashboard: portfolio KPIs, risk mix, decision breakdown."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.utils import api_client
from frontend.utils.theme import apply_theme, page_header

st.set_page_config(page_title="Executive Dashboard", page_icon="📊", layout="wide")
apply_theme()
page_header("Executive Dashboard", "Portfolio-level KPIs and risk mix")

if not api_client.is_authenticated():
    st.warning("Please sign in on the home page first.")
    st.stop()

days = st.slider("Window (days)", 1, 180, 30)

try:
    stats = api_client.monitoring_stats(days=days)
    dist = api_client.prediction_distribution(days=days)
except api_client.ApiError as exc:
    st.error(f"Could not load dashboard data: {exc}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Predictions scored", f"{stats['n_predictions']:,}")
col2.metric("Avg. predicted PD", f"{(stats['avg_predicted_probability'] or 0):.1%}")
col3.metric("Avg. latency", f"{(stats['avg_latency_ms'] or 0):.1f} ms")
approved = dist.get("decision_counts", {}).get("approved", 0)
total = max(dist.get("n", 1), 1)
col4.metric("Approval rate", f"{approved / total:.1%}")

st.divider()

c1, c2 = st.columns(2)

with c1:
    st.subheader("Predicted default probability distribution")
    if dist.get("histogram"):
        hist = dist["histogram"]
        fig = go.Figure(
            go.Bar(
                x=[f"{h['bin_start']:.0%}-{h['bin_end']:.0%}" for h in hist],
                y=[h["count"] for h in hist],
                marker_color="#2563eb",
            )
        )
        fig.update_layout(template="plotly_dark", height=380, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions recorded yet in this window.")

with c2:
    st.subheader("Risk band mix")
    risk_counts = dist.get("risk_band_counts", {})
    if risk_counts:
        fig = px.pie(
            names=list(risk_counts.keys()),
            values=list(risk_counts.values()),
            color=list(risk_counts.keys()),
            color_discrete_map={
                "very_low": "#4ade80",
                "low": "#34d399",
                "medium": "#facc15",
                "high": "#fb923c",
                "very_high": "#f87171",
            },
            hole=0.45,
        )
        fig.update_layout(template="plotly_dark", height=380, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions recorded yet in this window.")

st.subheader("Loan approval analytics")
decision_counts = dist.get("decision_counts", {})
if decision_counts:
    fig = go.Figure(
        go.Bar(
            x=list(decision_counts.keys()),
            y=list(decision_counts.values()),
            marker_color=["#4ade80" if k == "approved" else "#f87171" for k in decision_counts.keys()],
        )
    )
    fig.update_layout(template="plotly_dark", height=320, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No decisions recorded yet.")
