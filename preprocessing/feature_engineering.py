"""
Domain-driven feature engineering for credit risk.

Design decision: rather than relying purely on automated feature synthesis
(featuretools-style deep feature synthesis is overkill for a single flat
table like this), we hand-derive ratio/interaction features that underwriters
and risk models actually use in practice (payment-to-income, utilization
bands, credit history depth). This keeps every derived feature explainable
in a SHAP report — "loan_to_income_ratio pushed risk up" is a sentence a
credit committee can act on; an opaque auto-generated feature is not.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.logging import get_logger

logger = get_logger(__name__)


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless domain feature derivations (no fit-time leakage risk)."""

    def fit(self, X: pd.DataFrame, y=None) -> CreditFeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        eps = 1e-6

        X["loan_to_income_ratio"] = X["loan_amount"] / (X["annual_income"] + eps)
        X["monthly_payment_estimate"] = (
            X["loan_amount"] * (1 + X["interest_rate"] / 100) / X["loan_term_months"].clip(lower=1)
        )
        X["payment_to_income_ratio"] = (X["monthly_payment_estimate"] * 12) / (X["annual_income"] + eps)

        X["total_liquid_assets"] = X["savings_balance"] + X["checking_balance"]
        X["liquidity_to_loan_ratio"] = X["total_liquid_assets"] / (X["loan_amount"] + eps)

        X["credit_utilization_band"] = pd.cut(
            X["revolving_utilization"],
            bins=[-0.01, 0.1, 0.3, 0.5, 0.75, 1.01],
            labels=["very_low", "low", "moderate", "high", "very_high"],
        ).astype(str)

        X["credit_score_band"] = pd.cut(
            X["credit_score"],
            bins=[0, 579, 669, 739, 799, 850],
            labels=["poor", "fair", "good", "very_good", "excellent"],
        ).astype(str)

        X["delinquency_severity"] = X["num_delinquencies_2yrs"] + 2 * X["num_derogatory_marks"]
        X["has_recent_delinquency"] = (X["months_since_last_delinquency"] <= 24).astype(int)
        X["inquiries_per_open_account"] = X["num_credit_inquiries_6m"] / (X["num_open_accounts"] + 1)

        X["income_per_dependent"] = X["annual_income"] / (X["num_dependents"] + 1)
        X["is_stable_employment"] = (X["employment_length_years"] >= 2).astype(int)
        X["is_homeowner"] = (X["home_ownership"] == "OWN").astype(int)

        X["credit_history_depth"] = X["num_open_accounts"] + X["num_credit_inquiries_6m"]
        X["available_credit"] = X["total_credit_limit"] * (1 - X["revolving_utilization"])

        # Interaction: high utilization matters far more at low credit scores.
        X["utilization_x_low_score"] = X["revolving_utilization"] * (X["credit_score"] < 650).astype(int)

        logger.info("Feature engineering applied", extra={"context": {"n_columns": X.shape[1]}})
        return X

    def get_engineered_feature_names(self) -> list[str]:
        return [
            "loan_to_income_ratio",
            "monthly_payment_estimate",
            "payment_to_income_ratio",
            "total_liquid_assets",
            "liquidity_to_loan_ratio",
            "credit_utilization_band",
            "credit_score_band",
            "delinquency_severity",
            "has_recent_delinquency",
            "inquiries_per_open_account",
            "income_per_dependent",
            "is_stable_employment",
            "is_homeowner",
            "credit_history_depth",
            "available_credit",
            "utilization_x_low_score",
        ]
