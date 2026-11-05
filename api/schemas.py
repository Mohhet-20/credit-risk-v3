"""
Pydantic request/response schemas for the API.

Design decision: `CreditApplicationRequest` mirrors `feature_store.raw_features.SCHEMA`
field-for-field with explicit `Field(ge=..., le=...)` bounds pulled from the
same `RANGE_BOUNDS` used by the data-quality validator, so invalid requests
are rejected at the FastAPI validation layer (HTTP 422, before touching the
model) rather than silently propagating garbage into the feature pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from feature_store.raw_features import SCHEMA

# --- Auth ------------------------------------------------------------------
# Password policy: length is enforced by the `Field(min_length=8, ...)`
# bound on each schema below; complexity (upper/lower/digit/special char)
# and a small common-password blocklist are enforced here so both the
# public self-service schema (`UserCreate`) and the admin-only schema
# (`AdminUserCreate`) share exactly one policy rather than risking drift
# between two hand-copied validators.
_COMMON_PASSWORDS = {
    "password",
    "password1",
    "password123",
    "12345678",
    "123456789",
    "qwerty123",
    "letmein1",
    "admin1234",
    "welcome123",
    "changeme123",
}


def _validate_password_strength(password: str) -> str:
    if password.lower() in _COMMON_PASSWORDS:
        raise ValueError("Password is too common; choose a less predictable password.")
    if not any(c.isupper() for c in password):
        raise ValueError("Password must contain at least one uppercase letter.")
    if not any(c.islower() for c in password):
        raise ValueError("Password must contain at least one lowercase letter.")
    if not any(c.isdigit() for c in password):
        raise ValueError("Password must contain at least one digit.")
    if not any(not c.isalnum() for c in password):
        raise ValueError("Password must contain at least one special character.")
    return password


# Self-service roles only -- "Read-Only User" (viewer) and "Business User"
# (loan_officer) in the design's naming. Privileged roles (super_admin, admin,
# ml_engineer, risk_analyst, auditor) can NEVER be requested through public
# self-registration; see `services/auth_service.py::AuthService.register`'s
# `SELF_SERVICE_ROLES` gate and `AdminUserCreate` below for how privileged
# accounts actually get created. This closes what was previously a
# privilege-escalation hole: `UserCreate.role` used to accept "admin"
# directly from an unauthenticated caller.
class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: Literal["viewer", "loan_officer"] = "viewer"

    @field_validator("password")
    @classmethod
    def _check_password_strength(cls, v: str) -> str:
        return _validate_password_strength(v)


# Used only by the admin-gated `POST /admin/users` -- the one place a
# privileged role can be assigned at creation time, by an already-authenticated
# admin/super_admin (enforced by the route's `require_role("admin")`, not by
# this schema, which only constrains which *values* are well-formed).
class AdminUserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: Literal["super_admin", "admin", "ml_engineer", "risk_analyst", "auditor", "loan_officer", "viewer"]

    @field_validator("password")
    @classmethod
    def _check_password_strength(cls, v: str) -> str:
        return _validate_password_strength(v)


class UserOut(BaseModel):
    id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class LoginResponse(BaseModel):
    """`POST /auth/login`'s response shape. When the account does NOT have
    MFA enabled (the default), this is behaviorally identical to
    `TokenResponse` -- `access_token`/`refresh_token` populated,
    `mfa_required=False`. When MFA IS enabled, `access_token`/`refresh_token`
    are `None` and `mfa_challenge_token` must be exchanged (along with a TOTP
    or backup code) at `POST /auth/mfa/verify-login` to actually obtain
    tokens -- see `docs/SECURITY.md` for the full flow."""

    access_token: str | None = None
    refresh_token: str | None = None
    token_type: str = "bearer"
    mfa_required: bool = False
    mfa_challenge_token: str | None = None


class RefreshRequest(BaseModel):
    refresh_token: str


# --- MFA (TOTP) --------------------------------------------------------------
class MfaEnrollResponse(BaseModel):
    secret: str
    otpauth_uri: str


class MfaConfirmRequest(BaseModel):
    code: str = Field(min_length=6, max_length=6)


class MfaConfirmResponse(BaseModel):
    backup_codes: list[str]


class MfaDisableRequest(BaseModel):
    password: str


class MfaVerifyLoginRequest(BaseModel):
    mfa_challenge_token: str
    code: str = Field(min_length=6, max_length=20)  # 6-digit TOTP or an "XXXXXXXX-XXXXXXXX" backup code (17 chars)


# --- Prediction --------------------------------------------------------------
_bounds = SCHEMA.RANGE_BOUNDS


class CreditApplicationRequest(BaseModel):
    customer_id: str | None = None

    age: int = Field(ge=_bounds["age"][0], le=_bounds["age"][1])
    annual_income: float = Field(ge=_bounds["annual_income"][0], le=_bounds["annual_income"][1])
    employment_length_years: float = Field(ge=0, le=60)
    loan_amount: float = Field(ge=_bounds["loan_amount"][0], le=_bounds["loan_amount"][1])
    loan_term_months: int = Field(ge=1, le=360)
    interest_rate: float = Field(ge=0, le=60)
    credit_score: int = Field(ge=300, le=850)
    debt_to_income_ratio: float = Field(ge=0, le=100)
    num_open_accounts: int = Field(ge=0, le=100)
    num_credit_inquiries_6m: int = Field(ge=0, le=50)
    revolving_utilization: float = Field(ge=0, le=2)
    num_delinquencies_2yrs: int = Field(ge=0, le=50)
    months_since_last_delinquency: int = Field(ge=0, le=999)
    num_derogatory_marks: int = Field(ge=0, le=50)
    total_credit_limit: float = Field(ge=_bounds["total_credit_limit"][0], le=_bounds["total_credit_limit"][1])
    savings_balance: float = Field(ge=_bounds["savings_balance"][0], le=_bounds["savings_balance"][1])
    checking_balance: float = Field(ge=_bounds["checking_balance"][0], le=_bounds["checking_balance"][1])
    num_dependents: int = Field(ge=0, le=20)
    years_at_residence: float = Field(ge=0, le=80)

    home_ownership: Literal["OWN", "MORTGAGE", "RENT", "OTHER"]
    loan_purpose: Literal[
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "major_purchase",
        "medical",
        "small_business",
        "car",
        "other",
    ]
    employment_status: Literal["employed", "self_employed", "unemployed", "retired"]
    education_level: Literal["high_school", "bachelors", "masters", "doctorate", "other"]
    marital_status: Literal["single", "married", "divorced", "widowed"]
    state: str


class AdverseActionReason(BaseModel):
    """One principal reason for a decline (ECOA / Reg B notice of adverse action)."""

    rank: int
    code: str
    reason: str
    contribution: float
    features: list[str]


class PredictionResponse(BaseModel):
    prediction_id: str
    customer_id: str | None
    predicted_probability: float
    credit_score: int = Field(description="Scorecard points (PDO scaling: 600 = 50:1 good:bad odds, +20 doubles odds)")
    risk_band: str
    decision: str
    decision_threshold: float
    adverse_action_reasons: list[AdverseActionReason] = Field(
        default_factory=list, description="Populated for declined applications only, riskiest reason first."
    )
    model_name: str
    model_version: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    applications: list[CreditApplicationRequest]


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    n_processed: int


class BatchJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    n_total: int
    n_processed: int
    result_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Explainability ------------------------------------------------------------
class ExplanationContribution(BaseModel):
    feature: str
    shap_value: float
    feature_value: Any


class ExplanationResponse(BaseModel):
    prediction_id: str | None = None
    base_value: float
    predicted_value: float
    contributions: list[ExplanationContribution]
    plain_english: str
    adverse_action_reasons: list[AdverseActionReason] = Field(default_factory=list)


class CounterfactualChange(BaseModel):
    feature: str
    from_value: float
    to_value: float
    resulting_probability: float


class CounterfactualResponse(BaseModel):
    achieved: bool
    original_probability: float
    final_probability: float
    n_steps: int
    changes: list[CounterfactualChange]


class LimeContribution(BaseModel):
    rule: str
    weight: float


class LimeExplanationResponse(BaseModel):
    method: Literal["lime"] = "lime"
    intercept: float
    local_prediction: float | None
    contributions: list[LimeContribution]


# --- Models / monitoring --------------------------------------------------------
class ModelVersionOut(BaseModel):
    version: str
    model_name: str
    registered_at: str
    decision_threshold: float | None = None
    test_metrics: dict | None = None
    is_champion: bool = False


class PromoteModelRequest(BaseModel):
    version: str


class MonitoringStats(BaseModel):
    n_predictions: int
    avg_predicted_probability: float | None
    avg_latency_ms: float | None
    window_days: int


class HealthResponse(BaseModel):
    status: str
    champion_model: str | None
    champion_version: str | None


# --- Admin: users / audit ------------------------------------------------------
class AdminUserOut(BaseModel):
    id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    failed_login_attempts: int
    locked_until: datetime | None
    last_login_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class UpdateUserStatusRequest(BaseModel):
    is_active: bool


class UpdateUserRoleRequest(BaseModel):
    role: Literal["super_admin", "admin", "ml_engineer", "risk_analyst", "auditor", "loan_officer", "viewer"]


class AuditLogOut(BaseModel):
    id: str
    user_id: str | None
    action: str
    resource: str | None
    details: dict | None
    ip_address: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PageMeta(BaseModel):
    page: int
    size: int
    total: int
    total_pages: int


class AdminUserPage(BaseModel):
    items: list[AdminUserOut]
    page: int
    size: int
    total: int
    total_pages: int


class AuditLogPage(BaseModel):
    items: list[AuditLogOut]
    page: int
    size: int
    total: int
    total_pages: int


# --- Business Rule Engine ---------------------------------------------------
class BusinessRuleCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    rule_type: Literal["approval", "rejection", "manual_review"]
    condition: dict[str, Any]
    priority: int = Field(default=100, ge=1, le=1000)


class BusinessRuleUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = None
    rule_type: Literal["approval", "rejection", "manual_review"] | None = None
    condition: dict[str, Any] | None = None
    priority: int | None = Field(default=None, ge=1, le=1000)


class BusinessRuleOut(BaseModel):
    id: str
    name: str
    description: str | None
    rule_type: str
    condition: dict[str, Any]
    priority: int
    is_active: bool
    version: int
    created_by: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RuleSimulateRequest(BaseModel):
    contexts: list[dict[str, Any]] = Field(min_length=1)
    rule_ids: list[str] | None = None


# --- Enterprise Workflow Engine ---------------------------------------------
class WorkflowExecuteResponse(BaseModel):
    execution_id: str
    status: str
    final_decision: str | None
    prediction: dict[str, Any]
    rule_evaluation: dict[str, Any] | None
    needs_manual_review: bool
    steps: list[dict[str, Any]]


class WorkflowExecutionOut(BaseModel):
    id: str
    customer_id: str | None
    prediction_id: str | None
    status: str
    final_decision: str | None
    steps: list[dict[str, Any]]
    started_at: datetime
    completed_at: datetime | None
    requested_by: str | None

    model_config = ConfigDict(from_attributes=True)


# --- Enterprise Configuration Center ----------------------------------------
class SystemConfigOut(BaseModel):
    key: str
    value: Any
    category: str
    description: str | None
    is_default: bool
    updated_at: str | None


class SystemConfigSetRequest(BaseModel):
    value: Any
    category: str = Field(min_length=1, max_length=32)
    description: str | None = None
