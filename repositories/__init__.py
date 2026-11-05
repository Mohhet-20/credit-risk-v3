"""
Repository layer: encapsulates all direct SQLAlchemy `Session` access behind
a typed, per-aggregate interface.

Design decision (Repository Pattern): routers and services depend on these
classes, never on `sqlalchemy.orm.Session` queries directly. This is the
single seam a future swap (a different ORM, a different database, a caching
layer in front of reads, or splitting a table out to its own microservice)
would go through — every call site that matters is already funneled through
one of these classes rather than scattered `select(...)` statements across
the API layer. It also makes services trivially unit-testable: a test can
construct a repository against an in-memory SQLite session with zero FastAPI
machinery involved.

This replaces the earlier flat `database/crud.py` module (kept working
throughout for a schema this size, but a real duplication risk
once a service layer needs the same queries) -- every function that used to
live there now has a home as a method on the repository for its aggregate.
"""

from __future__ import annotations

from repositories.audit_log_repository import AuditLogRepository
from repositories.data_quality_repository import DataQualityReportRepository
from repositories.model_version_repository import ModelVersionRepository
from repositories.prediction_repository import PredictionRepository
from repositories.user_repository import UserRepository

__all__ = [
    "AuditLogRepository",
    "DataQualityReportRepository",
    "ModelVersionRepository",
    "PredictionRepository",
    "UserRepository",
]
