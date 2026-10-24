"""
Raw feature schema — the contract for data entering the platform.

This is the single source of truth for "what a raw credit application looks
like" before any engineering happens. The API request schema (Pydantic),
the training data loader, and the data-quality validator all import from
here so a schema change (adding/renaming a raw field) only has to happen in
one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.constants import ID_COLUMNS, RAW_CATEGORICAL_FEATURES, RAW_NUMERIC_FEATURES, TARGET_COLUMN


@dataclass(frozen=True)
class RawFeatureSchema:
    numeric: tuple[str, ...] = tuple(RAW_NUMERIC_FEATURES)
    categorical: tuple[str, ...] = tuple(RAW_CATEGORICAL_FEATURES)
    identifiers: tuple[str, ...] = tuple(ID_COLUMNS)
    target: str = TARGET_COLUMN

    @property
    def all_features(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)

    # Sensible min/max bounds used for schema validation (data_quality module)
    # and for the API's request validation. Bounds are intentionally generous —
    # they exist to catch obviously corrupt input, not to enforce underwriting
    # policy (that's the model's job).
    RANGE_BOUNDS = {
        "age": (18, 100),
        "annual_income": (0, 5_000_000),
        "employment_length_years": (0, 60),
        "loan_amount": (100, 1_000_000),
        "loan_term_months": (1, 360),
        "interest_rate": (0, 60),
        "credit_score": (300, 850),
        "debt_to_income_ratio": (0, 100),
        "num_open_accounts": (0, 100),
        "num_credit_inquiries_6m": (0, 50),
        "revolving_utilization": (0, 2),  # allow slight overage above 1.0 (real-world data does happen)
        "num_delinquencies_2yrs": (0, 50),
        "months_since_last_delinquency": (0, 999),
        "num_derogatory_marks": (0, 50),
        "total_credit_limit": (0, 5_000_000),
        "savings_balance": (0, 50_000_000),
        "checking_balance": (0, 50_000_000),
        "num_dependents": (0, 20),
        "years_at_residence": (0, 80),
    }

    CATEGORICAL_ALLOWED_VALUES = {
        "home_ownership": {"OWN", "MORTGAGE", "RENT", "OTHER"},
        "loan_purpose": {
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "major_purchase",
            "medical",
            "small_business",
            "car",
            "other",
        },
        "employment_status": {"employed", "self_employed", "unemployed", "retired"},
        "education_level": {"high_school", "bachelors", "masters", "doctorate", "other"},
        "marital_status": {"single", "married", "divorced", "widowed"},
    }


SCHEMA = RawFeatureSchema()
