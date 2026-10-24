"""Customer search: browse and filter prediction history."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from frontend.utils import api_client
from frontend.utils.theme import apply_theme, page_header

st.set_page_config(page_title="Customer Search", page_icon="🔎", layout="wide")
apply_theme()
page_header("Customer Search", "Browse and filter prediction history")

if not api_client.is_authenticated():
    st.warning("Please sign in on the home page first.")
    st.stop()

col1, col2 = st.columns([2, 1])
customer_id_filter = col1.text_input("Filter by customer ID (exact match, optional)")
limit = col2.number_input("Max rows", 10, 2000, 200)

params = {"limit": limit}
if customer_id_filter:
    params["customer_id"] = customer_id_filter

try:
    resp = api_client._request(
        "GET",
        f"{api_client.API_BASE_URL}{api_client.API_PREFIX}/predict/history",
        params=params,
        headers=api_client._headers(),
        timeout=15,
    )
    records = api_client._handle(resp)
except api_client.ApiError as exc:
    st.error(str(exc))
    records = []

if records:
    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True, height=500)

    c1, c2, c3 = st.columns(3)
    c1.metric("Records shown", len(df))
    c2.metric("Approval rate", f"{(df['decision'] == 'approved').mean():.1%}")
    c3.metric("Avg. predicted PD", f"{df['predicted_probability'].mean():.1%}")

    st.download_button(
        "Download as CSV", data=df.to_csv(index=False), file_name="prediction_history.csv", mime="text/csv"
    )
else:
    st.info("No prediction history found — score some applicants on the Predict page first.")
