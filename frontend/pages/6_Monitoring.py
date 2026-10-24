"""Monitoring: feature/target drift, latency, and error-rate visibility."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.utils import api_client
from frontend.utils.theme import apply_theme, page_header

st.set_page_config(page_title="Monitoring", page_icon="📡", layout="wide")
apply_theme()
page_header("Monitoring", "Drift detection, latency, and prediction volume")

if not api_client.is_authenticated():
    st.warning("Please sign in on the home page first.")
    st.stop()

st.subheader("Latency")
try:
    latency = api_client.latency_stats()
    if latency.get("n", 0) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("p50", f"{latency['p50_ms']:.1f} ms")
        c2.metric("p95", f"{latency['p95_ms']:.1f} ms")
        c3.metric("p99", f"{latency['p99_ms']:.1f} ms")
        c4.metric("max", f"{latency['max_ms']:.1f} ms")
    else:
        st.info("No latency data recorded yet.")
except api_client.ApiError as exc:
    st.error(str(exc))

st.divider()
st.subheader("Feature & target drift (vs. training reference window)")
st.caption("Population Stability Index (PSI): < 0.10 none · 0.10–0.25 moderate · > 0.25 significant")

if st.button("Run drift detection"):
    try:
        with st.spinner("Comparing reference vs. current data distributions..."):
            drift = api_client.drift_report()
        if "error" in drift:
            st.warning(drift["error"])
        else:
            severity_color = {"none": "🟢", "moderate": "🟡", "significant": "🔴"}
            st.markdown(
                f"**Overall severity:** {severity_color.get(drift['overall_severity'], '')} "
                f"{drift['overall_severity'].title()}"
            )
            fd = pd.DataFrame(drift["feature_drift"])
            if len(fd):
                fig = px.bar(
                    fd,
                    x="feature",
                    y="psi",
                    color="severity",
                    color_discrete_map={"none": "#4ade80", "moderate": "#facc15", "significant": "#f87171"},
                )
                fig.update_layout(template="plotly_dark", height=450, margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(fd, use_container_width=True)
    except api_client.ApiError as exc:
        st.error(str(exc))
