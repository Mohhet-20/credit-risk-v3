"""Shared applicant-input form, used by both the Predict and Explainability pages."""

from __future__ import annotations

import streamlit as st

DEFAULTS = {
    "customer_id": "DEMO-0001",
    "age": 35,
    "annual_income": 55000.0,
    "employment_length_years": 3.0,
    "loan_amount": 12000.0,
    "loan_term_months": 36,
    "interest_rate": 13.5,
    "credit_score": 660,
    "debt_to_income_ratio": 22.0,
    "num_open_accounts": 7,
    "num_credit_inquiries_6m": 2,
    "revolving_utilization": 0.35,
    "num_delinquencies_2yrs": 0,
    "months_since_last_delinquency": 999,
    "num_derogatory_marks": 0,
    "total_credit_limit": 20000.0,
    "savings_balance": 3000.0,
    "checking_balance": 1500.0,
    "num_dependents": 1,
    "years_at_residence": 3.0,
    "home_ownership": "MORTGAGE",
    "loan_purpose": "debt_consolidation",
    "employment_status": "employed",
    "education_level": "bachelors",
    "marital_status": "married",
    "state": "CA",
}

_LOAN_PURPOSES = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "medical",
    "small_business",
    "car",
    "other",
]
_STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]


def application_form(key_prefix: str) -> dict:
    d = DEFAULTS
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Applicant**")
        customer_id = st.text_input("Customer ID", d["customer_id"], key=f"{key_prefix}_cid")
        age = st.number_input("Age", 18, 100, d["age"], key=f"{key_prefix}_age")
        annual_income = st.number_input(
            "Annual income ($)", 0.0, 5_000_000.0, d["annual_income"], step=1000.0, key=f"{key_prefix}_income"
        )
        employment_status = st.selectbox(
            "Employment status",
            ["employed", "self_employed", "unemployed", "retired"],
            index=0,
            key=f"{key_prefix}_empstat",
        )
        employment_length_years = st.number_input(
            "Years employed", 0.0, 60.0, d["employment_length_years"], key=f"{key_prefix}_emplen"
        )
        education_level = st.selectbox(
            "Education", ["high_school", "bachelors", "masters", "doctorate", "other"], index=1, key=f"{key_prefix}_edu"
        )
        marital_status = st.selectbox(
            "Marital status", ["single", "married", "divorced", "widowed"], index=1, key=f"{key_prefix}_marital"
        )
        num_dependents = st.number_input("Dependents", 0, 20, d["num_dependents"], key=f"{key_prefix}_dependents")

    with col2:
        st.markdown("**Loan request**")
        loan_amount = st.number_input(
            "Loan amount ($)", 100.0, 1_000_000.0, d["loan_amount"], step=500.0, key=f"{key_prefix}_amt"
        )
        loan_term_months = st.selectbox("Term (months)", [12, 24, 36, 48, 60, 72], index=2, key=f"{key_prefix}_term")
        interest_rate = st.number_input("Interest rate (%)", 0.0, 60.0, d["interest_rate"], key=f"{key_prefix}_rate")
        loan_purpose = st.selectbox("Purpose", _LOAN_PURPOSES, index=0, key=f"{key_prefix}_purpose")
        home_ownership = st.selectbox(
            "Home ownership", ["OWN", "MORTGAGE", "RENT", "OTHER"], index=1, key=f"{key_prefix}_home"
        )
        state = st.selectbox("State", _STATES, index=0, key=f"{key_prefix}_state")
        years_at_residence = st.number_input(
            "Years at residence", 0.0, 80.0, d["years_at_residence"], key=f"{key_prefix}_resid"
        )

    with col3:
        st.markdown("**Credit profile**")
        credit_score = st.slider("Credit score", 300, 850, d["credit_score"], key=f"{key_prefix}_score")
        debt_to_income_ratio = st.number_input(
            "Debt-to-income (%)", 0.0, 100.0, d["debt_to_income_ratio"], key=f"{key_prefix}_dti"
        )
        revolving_utilization = st.slider(
            "Revolving utilization", 0.0, 1.5, d["revolving_utilization"], key=f"{key_prefix}_util"
        )
        num_open_accounts = st.number_input(
            "Open accounts", 0, 100, d["num_open_accounts"], key=f"{key_prefix}_openacc"
        )
        num_credit_inquiries_6m = st.number_input(
            "Inquiries (6mo)", 0, 50, d["num_credit_inquiries_6m"], key=f"{key_prefix}_inq"
        )
        num_delinquencies_2yrs = st.number_input(
            "Delinquencies (2yr)", 0, 50, d["num_delinquencies_2yrs"], key=f"{key_prefix}_delinq"
        )
        months_since_last_delinquency = st.number_input(
            "Months since last delinquency",
            0,
            999,
            d["months_since_last_delinquency"],
            key=f"{key_prefix}_msld",
            help="Leave at 999 if there have been no delinquencies in the past 2 years.",
        )
        num_derogatory_marks = st.number_input(
            "Derogatory marks", 0, 50, d["num_derogatory_marks"], key=f"{key_prefix}_derog"
        )
        total_credit_limit = st.number_input(
            "Total credit limit ($)", 0.0, 5_000_000.0, d["total_credit_limit"], step=500.0, key=f"{key_prefix}_limit"
        )
        savings_balance = st.number_input(
            "Savings balance ($)", 0.0, 50_000_000.0, d["savings_balance"], step=100.0, key=f"{key_prefix}_savings"
        )
        checking_balance = st.number_input(
            "Checking balance ($)", 0.0, 50_000_000.0, d["checking_balance"], step=100.0, key=f"{key_prefix}_checking"
        )

    return {
        "customer_id": customer_id,
        "age": int(age),
        "annual_income": float(annual_income),
        "employment_length_years": float(employment_length_years),
        "loan_amount": float(loan_amount),
        "loan_term_months": int(loan_term_months),
        "interest_rate": float(interest_rate),
        "credit_score": int(credit_score),
        "debt_to_income_ratio": float(debt_to_income_ratio),
        "num_open_accounts": int(num_open_accounts),
        "num_credit_inquiries_6m": int(num_credit_inquiries_6m),
        "revolving_utilization": float(revolving_utilization),
        "num_delinquencies_2yrs": int(num_delinquencies_2yrs),
        "months_since_last_delinquency": int(months_since_last_delinquency),
        "num_derogatory_marks": int(num_derogatory_marks),
        "total_credit_limit": float(total_credit_limit),
        "savings_balance": float(savings_balance),
        "checking_balance": float(checking_balance),
        "num_dependents": int(num_dependents),
        "years_at_residence": float(years_at_residence),
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "employment_status": employment_status,
        "education_level": education_level,
        "marital_status": marital_status,
        "state": state,
    }
