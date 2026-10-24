"""
SQLAlchemy ORM models.

Design decision: `DATABASE_URL` (see `configs/settings.py`) defaults to a
local SQLite file so the platform runs with zero external setup, but every
model here uses only ANSI-portable column types (no Postgres-only JSONB
operators beyond the generic `JSON` type, which SQLAlchemy transparently
maps to native JSONB on Postgres and a TEXT-backed JSON shim on SQLite) —
so the exact same schema and CRUD code work unmodified against either
backend. Swapping to Postgres in production is purely a connection-string
change (`DATABASE_URL=postgresql://...` in `.env` / Docker Compose), not a
model rewrite.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="viewer", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Account-lockout fields (additive, nullable/defaulted -- see
    # docs/ARCHITECTURE.md "Account lockout" for why these are plain columns
    # rather than an Alembic migration).
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Refresh-token rotation (additive -- see core/security.py::create_refresh_token
    # and services/auth_service.py::AuthService.refresh). Bumped on every
    # successful refresh; a refresh token encoding an older version is
    # rejected as a rotated/replayed token.
    token_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # TOTP-based MFA (additive, nullable/defaulted -- see
    # core/security.py's "MFA (TOTP)" section and
    # services/auth_service.py::AuthService.enroll_mfa/confirm_mfa/disable_mfa).
    # `mfa_secret` is written at enrollment time but `mfa_enabled` stays
    # False (and login is unaffected) until `confirm_mfa` verifies a real
    # code from the user's authenticator app -- this prevents a user from
    # locking themselves out by enrolling with a secret they never actually
    # loaded into an app. `mfa_backup_codes` is a JSON-encoded list of
    # bcrypt-hashed single-use codes (never the plaintext -- see
    # `core/security.py::generate_backup_codes`'s docstring).
    mfa_secret: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    mfa_backup_codes: Mapped[str | None] = mapped_column(Text, nullable=True)

    predictions: Mapped[list[Prediction]] = relationship(back_populates="requested_by_user")
    audit_logs: Mapped[list[AuditLog]] = relationship(back_populates="user")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    customer_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    input_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    predicted_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_band: Mapped[str] = mapped_column(String(16), nullable=False)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)  # approved | rejected | manual_review
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=True)
    request_source: Mapped[str] = mapped_column(String(32), default="api")  # api | batch | streaming
    requested_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    requested_by_user: Mapped[User | None] = relationship(back_populates="predictions")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource: Mapped[str] = mapped_column(String(128), nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    user: Mapped[User | None] = relationship(back_populates="audit_logs")


class ModelVersionRecord(Base):
    """Mirrors the local ModelRegistry / MLflow registry into Postgres for fast
    dashboard queries — the registry filesystem/MLflow store remains the
    source of truth for the artifacts themselves."""

    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    version: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), default="None")  # None | Staging | Production | Archived
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    decision_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ExperimentRun(Base):
    """A lightweight local mirror of MLflow run metadata for dashboard listing
    without a live MLflow tracking server round-trip."""

    __tablename__ = "experiment_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_name: Mapped[str] = mapped_column(String(128), nullable=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class DataQualityReport(Base):
    """Stores the output of the data-quality / drift pipeline for time-series
    monitoring in the dashboard (drift-over-time charts)."""

    __tablename__ = "data_quality_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    report_type: Mapped[str] = mapped_column(String(32), nullable=False)  # schema | drift | quality
    summary: Mapped[dict] = mapped_column(JSON, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class BusinessRule(Base):
    """A single, DB-persisted, admin-editable business rule -- see
    `rules/engine.py` for evaluation semantics and `docs/ARCHITECTURE.md`'s
    "Business Rule Engine" section for the full design rationale.

    Design decision: `condition` is a small JSON expression (`{"field": ...,
    "operator": ..., "value": ...}` or a `{"all": [...]}`/`{"any": [...]}`
    composite of the same) rather than a free-text expression string
    (e.g. a Python `eval`-able snippet) -- this closes off arbitrary code
    execution entirely (a rule is data, never code) and keeps every rule
    both admin-editable through a form and mechanically simulatable/
    explainable, at the cost of not supporting arbitrarily exotic
    conditions -- an intentional, deliberate tradeoff for a regulated
    domain where "what exactly can this rule do" must be a closed,
    auditable question.

    `version` is bumped (not the row overwritten) on every edit, and the
    previous version's snapshot is preserved via `AuditLog` (action=
    "rule.update", details={"previous": {...}}) rather than a separate rule-
    history table -- reusing the audit trail that already exists rather
    than duplicating "who changed what, when" bookkeeping in a second
    place."""

    __tablename__ = "business_rules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    rule_type: Mapped[str] = mapped_column(String(32), nullable=False)  # approval | rejection | manual_review
    condition: Mapped[dict] = mapped_column(JSON, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False, index=True)  # lower runs first
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class WorkflowExecution(Base):
    """One row per run of the enterprise workflow engine (see
    `workflows/engine.py`) -- the standard credit-risk pipeline (validation
    -> data quality -> fraud check -> AI prediction -> business rules ->
    manual review -> decision -> report -> notification -> audit),
    expressed as an explicit, named, timed sequence of steps rather than
    an implicit call chain. `steps` is a JSON list of `{"name": str,
    "status": "completed"|"failed"|"skipped", "started_at": iso str,
    "completed_at": iso str, "duration_ms": float, "detail": str | None}`
    dicts -- this is what backs the "Request Timeline" / "Workflow
    Metrics" observability surfaces (`GET /workflow/executions/{id}`,
    `GET /monitoring/workflow-metrics`) without standing up a separate
    distributed-tracing system for a single-process pipeline that doesn't
    need one."""

    __tablename__ = "workflow_executions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    customer_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    prediction_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("predictions.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running", nullable=False)  # running|completed|failed
    final_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    steps: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    requested_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)


class SystemConfig(Base):
    """Enterprise Configuration Center: a simple, DB-persisted key/value
    store for everything that should be adjustable without a code deploy
    -- thresholds, feature flags, notification settings, workflow step
    toggles, model-selection preferences. Design decision: one generic
    table (`key` unique, `value` JSON, `category` for grouping in the
    admin UI) rather than a dedicated table per config domain -- the
    actual *number* of config keys this platform needs is small and
    growing slowly, so a dedicated `notification_settings` /
    `feature_flags` / `workflow_config` table each would be schema
    overhead for no real query-pattern benefit; `category` gives the same
    filtering ability a dedicated table would via `WHERE category = ...`.
    See `services/config_service.py`."""

    __tablename__ = "system_config"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    key: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    value: Mapped[dict] = mapped_column(JSON, nullable=False)
    category: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # rules|thresholds|notifications|feature_flags|workflow|model_selection
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
