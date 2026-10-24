"""Predict page: single-applicant scoring and batch CSV upload."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from frontend.utils import api_client
from frontend.utils.forms import application_form
from frontend.utils.theme import apply_theme, decision_html, page_header, risk_badge_html

st.set_page_config(page_title="Predict", page_icon="🎯", layout="wide")
apply_theme()
page_header("Predict", "Score a single applicant or a batch CSV upload")

if not api_client.is_authenticated():
    st.warning("Please sign in on the home page first.")
    st.stop()

tab_single, tab_batch = st.tabs(["Single applicant", "Batch (CSV upload)"])

with tab_single:
    application = application_form("predict")
    if st.button("Score applicant", type="primary"):
        try:
            result = api_client.predict(application)
            st.session_state["last_application"] = application
            st.session_state["last_result"] = result
        except api_client.ApiError as exc:
            st.error(str(exc))

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted default probability", f"{result['predicted_probability']:.1%}")
        c2.markdown(f"**Risk band**<br>{risk_badge_html(result['risk_band'])}", unsafe_allow_html=True)
        c3.markdown(f"**Decision**<br>{decision_html(result['decision'])}", unsafe_allow_html=True)
        c4.metric("Model", f"{result['model_name']}")

        if st.button("📄 Download PDF explainability report"):
            try:
                pdf_bytes = api_client.download_report(st.session_state["last_application"])
                st.download_button(
                    "Save report",
                    data=pdf_bytes,
                    file_name=f"credit_report_{result['prediction_id'][:8]}.pdf",
                    mime="application/pdf",
                )
            except api_client.ApiError as exc:
                st.error(str(exc))

with tab_batch:
    st.write(
        "Upload a CSV with one row per applicant. Columns must match the applicant schema "
        "(see the sidebar sample below)."
    )
    st.caption(
        "Required columns: customer_id, age, annual_income, employment_length_years, loan_amount, "
        "loan_term_months, interest_rate, credit_score, debt_to_income_ratio, num_open_accounts, "
        "num_credit_inquiries_6m, revolving_utilization, num_delinquencies_2yrs, "
        "months_since_last_delinquency, num_derogatory_marks, total_credit_limit, savings_balance, "
        "checking_balance, num_dependents, years_at_residence, home_ownership, loan_purpose, "
        "employment_status, education_level, marital_status, state"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(10), use_container_width=True)
        if st.button(f"Score {len(df)} applicants", type="primary"):
            applications = df.to_dict(orient="records")
            with st.spinner("Scoring batch..."):
                try:
                    batch_result = api_client.predict_batch(applications)
                    result_df = pd.DataFrame(batch_result["results"])
                    st.success(f"Scored {batch_result['n_processed']} applicants.")
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button(
                        "Download results CSV",
                        data=result_df.to_csv(index=False),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )
                except api_client.ApiError as exc:
                    st.error(str(exc))
