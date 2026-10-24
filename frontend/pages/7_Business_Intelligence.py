"""Business Intelligence: customer segmentation, anomaly detection, loan
structure recommendations, and a feature-importance timeline across
registered model versions."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from frontend.utils import api_client
from frontend.utils.forms import application_form
from frontend.utils.theme import apply_theme, page_header

st.set_page_config(page_title="Business Intelligence", page_icon="🧭", layout="wide")
apply_theme()
page_header("Business Intelligence", "Segmentation, anomaly detection, loan recommendations, and model insight")

if not api_client.is_authenticated():
    st.warning("Please sign in on the home page first.")
    st.stop()

tab_segment, tab_anomaly, tab_recommend, tab_timeline = st.tabs(
    ["Customer Segment", "Anomaly Check", "Loan Recommendations", "Feature Importance Timeline"]
)

with tab_segment:
    st.caption(
        "Cluster this applicant against a reference sample of the training population "
        "and surface a business-readable risk segment."
    )
    application = application_form("bi_segment")
    if st.button("Segment this applicant", type="primary", key="segment_btn"):
        try:
            result = api_client.segment(application)
            c1, c2 = st.columns(2)
            c1.metric("Segment", result["label"])
            c2.metric("Reference sample size in segment", result["segment_size_in_reference_sample"])
            c3, c4, c5 = st.columns(3)
            c3.metric("Segment avg. credit score", result["segment_avg_credit_score"])
            c4.metric("Segment avg. utilization", f"{result['segment_avg_utilization']:.1%}")
            c5.metric("Segment avg. DTI", f"{result['segment_avg_debt_to_income']:.1f}%")
        except api_client.ApiError as exc:
            st.error(str(exc))

with tab_anomaly:
    st.caption(
        "Flags applications that look statistically unusual relative to the training "
        "population — useful as a fraud/data-entry-error signal, independent of credit risk."
    )
    application = application_form("bi_anomaly")
    if st.button("Check for anomalies", type="primary", key="anomaly_btn"):
        try:
            result = api_client.anomaly(application)
            if result["is_anomaly"]:
                st.error(f"⚠️ Flagged as anomalous — anomaly score {result['anomaly_score']:.2f}")
            else:
                st.success(f"No anomaly detected — anomaly score {result['anomaly_score']:.2f}")
            st.write("Most deviant fields (|z-score| vs. reference population):")
            st.dataframe(pd.DataFrame(result["top_deviant_features"]), use_container_width=True)
        except api_client.ApiError as exc:
            st.error(str(exc))

with tab_recommend:
    st.caption(
        "Searches realistic loan_amount / loan_term_months combinations and returns "
        "structures that would be approved, ranked by amount then monthly payment."
    )
    application = application_form("bi_recommend")
    if st.button("Get loan recommendations", type="primary", key="recommend_btn"):
        try:
            result = api_client.recommend(application)
            if result["n_options_found"] == 0:
                st.warning("No approved loan structure found within the searched amount/term grid.")
            else:
                st.dataframe(pd.DataFrame(result["recommendations"]), use_container_width=True)
        except api_client.ApiError as exc:
            st.error(str(exc))

with tab_timeline:
    st.caption(
        "Top drivers of predicted default risk for every model version ever registered — "
        "requires the risk_analyst or admin role."
    )
    if st.button("Load timeline", key="timeline_btn"):
        try:
            data = api_client.feature_importance_timeline()
            for entry in data["timeline"]:
                st.subheader(f"{entry['model_name']} — {entry['version']} ({entry['registered_at']})")
                if entry["top_features"]:
                    st.bar_chart(pd.DataFrame(entry["top_features"]).set_index("feature"))
                else:
                    st.info("This model type doesn't expose simple feature importances (e.g. a stacked ensemble).")
        except api_client.ApiError as exc:
            st.error(str(exc))
