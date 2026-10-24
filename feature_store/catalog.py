"""
Feature catalog — human-readable descriptions for every raw and engineered
feature, used by plain-English explanations, PDF reports, and the frontend's
feature-metadata endpoints.

Split from `registry.py`'s original generic placeholders into an explicit
per-field description map, because the whole point of a plain-English
explanation is that it reads like something a loan officer would actually
say — "Raw numeric field: credit_score" defeats that purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.constants import RAW_CATEGORICAL_FEATURES, RAW_NUMERIC_FEATURES
from preprocessing.feature_engineering import CreditFeatureEngineer

RAW_DESCRIPTIONS: dict[str, str] = {
    "age": "Applicant's age",
    "annual_income": "Annual income",
    "employment_length_years": "Years at current employer",
    "loan_amount": "Requested loan amount",
    "loan_term_months": "Requested loan term",
    "interest_rate": "Offered interest rate",
    "credit_score": "Credit score",
    "debt_to_income_ratio": "Debt-to-income ratio",
    "num_open_accounts": "Number of open credit accounts",
    "num_credit_inquiries_6m": "Credit inquiries in the last 6 months",
    "revolving_utilization": "Revolving credit utilization",
    "num_delinquencies_2yrs": "Delinquencies in the last 2 years",
    "months_since_last_delinquency": "Months since the applicant's last delinquency",
    "num_derogatory_marks": "Derogatory marks on credit history",
    "total_credit_limit": "Total available credit limit",
    "savings_balance": "Savings account balance",
    "checking_balance": "Checking account balance",
    "num_dependents": "Number of dependents",
    "years_at_residence": "Years at current residence",
    "home_ownership": "Home ownership status",
    "loan_purpose": "Stated loan purpose",
    "employment_status": "Employment status",
    "education_level": "Education level",
    "marital_status": "Marital status",
    "state": "State of residence",
}

ENGINEERED_DESCRIPTIONS: dict[str, str] = {
    "loan_to_income_ratio": "Loan amount relative to annual income",
    "monthly_payment_estimate": "Estimated monthly installment",
    "payment_to_income_ratio": "Estimated annual payment relative to income",
    "total_liquid_assets": "Total savings and checking balance",
    "liquidity_to_loan_ratio": "Liquid assets relative to the requested loan",
    "credit_utilization_band": "Credit utilization tier",
    "credit_score_band": "Credit score tier",
    "delinquency_severity": "Combined delinquency and derogatory-mark severity",
    "has_recent_delinquency": "Recent delinquency (within 24 months)",
    "inquiries_per_open_account": "Credit inquiries per open account",
    "income_per_dependent": "Income per dependent",
    "is_stable_employment": "Employment stability (2+ years)",
    "is_homeowner": "Homeownership",
    "credit_history_depth": "Depth of credit history (accounts + inquiries)",
    "available_credit": "Unused available credit",
    "utilization_x_low_score": "High utilization combined with a low credit score",
}


@dataclass(frozen=True)
class FeatureDescriptor:
    name: str
    source: str  # "raw" | "engineered"
    dtype: str  # "numeric" | "categorical"
    description: str


class FeatureCatalog:
    def __init__(self) -> None:
        self._descriptors: dict[str, FeatureDescriptor] = {}
        for name in RAW_NUMERIC_FEATURES:
            self._descriptors[name] = FeatureDescriptor(name, "raw", "numeric", RAW_DESCRIPTIONS.get(name, name))
        for name in RAW_CATEGORICAL_FEATURES:
            self._descriptors[name] = FeatureDescriptor(name, "raw", "categorical", RAW_DESCRIPTIONS.get(name, name))
        engineer = CreditFeatureEngineer()
        for name in engineer.get_engineered_feature_names():
            dtype = "categorical" if name.endswith("_band") else "numeric"
            self._descriptors[name] = FeatureDescriptor(
                name, "engineered", dtype, ENGINEERED_DESCRIPTIONS.get(name, name)
            )

    def get(self, name: str) -> FeatureDescriptor | None:
        # Handle derived column name suffixes (one-hot dummies, target-encoded columns).
        base_name = name
        for suffix in ("_target_enc",):
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
        if base_name in self._descriptors:
            return self._descriptors[base_name]
        # One-hot dummy columns look like "home_ownership_RENT" -> match on prefix.
        for original, descriptor in self._descriptors.items():
            if name.startswith(original + "_"):
                return descriptor
        return None

    def describe(self, name: str) -> str:
        descriptor = self.get(name)
        return descriptor.description if descriptor else name.replace("_", " ")

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([d.__dict__ for d in self._descriptors.values()])

    def all_names(self) -> list[str]:
        return list(self._descriptors.keys())


catalog = FeatureCatalog()
