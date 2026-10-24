"""Model comparison and governance: challenger table, version registry, promote/rollback."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.utils import api_client
from frontend.utils.theme import apply_theme, page_header

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
apply_theme()
page_header("Model Comparison", "Champion vs. challenger models, registry, and promotion")

if not api_client.is_authenticated():
    st.warning("Please sign in on the home page first.")
    st.stop()

try:
    comparison = api_client.model_comparison()
except api_client.ApiError as exc:
    comparison = []
    st.warning(f"No model comparison table available yet: {exc}")

if comparison:
    df = pd.DataFrame(comparison)
    st.subheader("Candidate models")
    st.dataframe(df, use_container_width=True)

    metric = st.selectbox("Metric", [c for c in ["roc_auc", "pr_auc", "f1", "brier_score"] if c in df.columns])
    fig = px.bar(df.sort_values(metric, ascending=False), x="model", y=metric, color="model")
    fig.update_layout(template="plotly_dark", height=400, margin=dict(t=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Model registry")
try:
    versions = api_client.model_versions()
except api_client.ApiError as exc:
    versions = []
    st.warning(f"Could not load model registry: {exc}")

if versions:
    for v in versions:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            badge = "🏆 CHAMPION" if v["is_champion"] else ""
            c1.markdown(f"**{v['version']}** {badge}")
            c1.caption(f"Model: {v['model_name']} · Registered: {v['registered_at']}")
            metrics = v.get("test_metrics") or {}
            c2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}" if metrics else "—")
            c3.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}" if metrics else "—")
            if not v["is_champion"]:
                if st.button(f"Promote {v['version']}", key=f"promote_{v['version']}"):
                    try:
                        api_client.promote_model(v["version"])
                        st.success("Promoted. Reloading...")
                        st.rerun()
                    except api_client.ApiError as exc:
                        st.error(str(exc))
else:
    st.info("No model versions registered yet — run the training pipeline first.")
