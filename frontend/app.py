"""
Streamlit frontend entrypoint — login/landing page.

Run with: `streamlit run frontend/app.py`
Requires the FastAPI backend running (default http://localhost:8000; override
with the API_BASE_URL env var).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from frontend.utils import api_client
from frontend.utils.theme import apply_theme, page_header

st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

page_header("Credit Risk Intelligence Platform", "Enterprise AI/ML platform for underwriting decisions")

with st.sidebar:
    st.markdown("### Navigation")
    st.caption("Use the pages listed below to explore the platform.")
    try:
        health = api_client.health()
        st.success(f"API online · champion: {health.get('champion_model') or 'none'}")
    except Exception:  # noqa: BLE001
        st.error("API unreachable — start the FastAPI backend (`uvicorn api.app:app`).")

if "mfa_challenge_token" in st.session_state:
    st.subheader("Two-factor verification")
    st.caption("Enter the 6-digit code from your authenticator app, or a backup code.")
    with st.form("mfa_form"):
        code = st.text_input("Verification code")
        submitted = st.form_submit_button("Verify", use_container_width=True)
        if submitted:
            try:
                api_client.verify_mfa_login(st.session_state["mfa_challenge_token"], code)
                st.session_state.pop("mfa_challenge_token", None)
                st.success("Signed in.")
                st.rerun()
            except api_client.ApiError as exc:
                st.error(str(exc))
    if st.button("Cancel"):
        st.session_state.pop("mfa_challenge_token", None)
        st.rerun()

elif not api_client.is_authenticated():
    st.subheader("Sign in")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
            if submitted:
                try:
                    result = api_client.login(username, password)
                    if result.get("mfa_required"):
                        st.session_state["mfa_challenge_token"] = result["mfa_challenge_token"]
                    else:
                        st.success("Signed in.")
                    st.rerun()
                except api_client.ApiError as exc:
                    st.error(str(exc))

    with tab_register:
        with st.form("register_form"):
            r_username = st.text_input("Username", key="reg_username")
            r_email = st.text_input("Email", key="reg_email")
            r_password = st.text_input("Password", type="password", key="reg_password")
            r_role = st.selectbox(
                "Role",
                ["loan_officer", "viewer"],
                key="reg_role",
                help="Self-service registration is limited to these roles. "
                "Risk analyst / admin accounts must be created by an existing admin.",
            )
            submitted = st.form_submit_button("Create account", use_container_width=True)
            if submitted:
                try:
                    api_client.register(r_username, r_email, r_password, r_role)
                    st.success("Account created — sign in on the Login tab.")
                except api_client.ApiError as exc:
                    st.error(str(exc))
else:
    st.success("You're signed in. Use the sidebar pages to navigate the platform:")
    st.markdown("""
        - **Executive Dashboard** — portfolio KPIs, approval trends, risk mix
        - **Predict** — score a single applicant or a batch CSV, download PDF reports
        - **Explainability** — SHAP breakdown, plain-English rationale, what-if simulator
        - **Model Comparison** — champion vs. challenger models, promote/rollback
        - **Customer Search** — browse and filter prediction history
        - **Monitoring** — drift detection, latency, prediction distribution
        """)
    if st.button("Sign out"):
        api_client.logout()
        st.rerun()
