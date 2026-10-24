"""
Centralized constants for the Credit Risk Intelligence Platform.

Keeping magic strings/numbers in one module prevents drift between the
training pipeline, the API, and the frontend — all three import from here
instead of redefining feature lists or thresholds independently.
"""

from __future__ import annotations

from enum import Enum


class RiskBand(str, Enum):
    """Business-facing risk tiers derived from the calibrated PD (probability of default)."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class LoanStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"


class ModelStage(str, Enum):
    """Mirrors MLflow Model Registry stages."""

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class UserRole(str, Enum):
    """The platform's 7-role RBAC model.

    Five roles form a linear ordinal hierarchy (see `core/security.py::RolePower`)
    where each role can do everything the roles below it can:
    VIEWER < LOAN_OFFICER < RISK_ANALYST < ADMIN < SUPER_ADMIN.

    AUDITOR and ML_ENGINEER are deliberately LATERAL, not inserted into that
    ordinal chain -- an auditor needs broad READ access (audit logs, reports,
    data-quality history, model list) without the WRITE power a risk_analyst
    or admin has, and an ml_engineer needs model-governance WRITE access
    (promote/rollback, trigger data-quality runs) without user-management
    power. Modeling them as "somewhere in the middle of the ordinal ladder"
    would either over-grant (an ordinal auditor above risk_analyst could
    also do everything risk_analyst can, including mutating endpoints) or
    under-grant them. Endpoints that need to admit a lateral role alongside
    an ordinal tier use `api/dependencies.py::require_roles()` (an explicit
    allow-list) rather than `require_role()` (ordinal "this tier or above").

    VIEWER and LOAN_OFFICER double as the spec's "Read-Only User" and
    "Business User" respectively -- kept as their original names rather than
    renamed, to avoid a breaking change across every existing role check,
    test, and stored user row.
    """

    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    ML_ENGINEER = "ml_engineer"
    RISK_ANALYST = "risk_analyst"
    AUDITOR = "auditor"
    LOAN_OFFICER = "loan_officer"  # "Business User"
    VIEWER = "viewer"  # "Read-Only User"


# Target column produced by the synthetic data generator / expected in real feeds.
TARGET_COLUMN = "default_flag"

# Columns that must never be used as model features (identifiers, leakage-prone fields).
ID_COLUMNS = ["customer_id", "application_id", "application_date"]

# Raw numeric features present at intake time.
RAW_NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_length_years",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "credit_score",
    "debt_to_income_ratio",
    "num_open_accounts",
    "num_credit_inquiries_6m",
    "revolving_utilization",
    "num_delinquencies_2yrs",
    "months_since_last_delinquency",
    "num_derogatory_marks",
    "total_credit_limit",
    "savings_balance",
    "checking_balance",
    "num_dependents",
    "years_at_residence",
]

# Raw categorical features present at intake time.
RAW_CATEGORICAL_FEATURES = [
    "home_ownership",
    "loan_purpose",
    "employment_status",
    "education_level",
    "marital_status",
    "state",
]

# Fair-lending exclusions: collected at intake (Reg B monitoring, fairness
# testing in `analytics/fairness.py`) but never fed to the model.
#   - age, marital_status: ECOA-protected bases (15 U.S.C. 1691(a)).
#   - state: geography is the textbook redlining proxy and carries no
#     underwriting signal a borrower-level feature doesn't already capture.
MODEL_EXCLUDED_FEATURES = ["age", "marital_status", "state"]

MODEL_NUMERIC_FEATURES = [f for f in RAW_NUMERIC_FEATURES if f not in MODEL_EXCLUDED_FEATURES]
MODEL_CATEGORICAL_FEATURES = [f for f in RAW_CATEGORICAL_FEATURES if f not in MODEL_EXCLUDED_FEATURES]

# Risk band cut points on predicted probability of default (post-calibration).
RISK_BAND_THRESHOLDS: dict[RiskBand, tuple[float, float]] = {
    RiskBand.VERY_LOW: (0.0, 0.05),
    RiskBand.LOW: (0.05, 0.15),
    RiskBand.MEDIUM: (0.15, 0.35),
    RiskBand.HIGH: (0.35, 0.60),
    RiskBand.VERY_HIGH: (0.60, 1.01),
}

RANDOM_STATE = 42
N_CV_FOLDS = 5

# Account lockout policy (see services/auth_service.py::AuthService.login).
MAX_FAILED_LOGIN_ATTEMPTS = 5
ACCOUNT_LOCKOUT_MINUTES = 15
