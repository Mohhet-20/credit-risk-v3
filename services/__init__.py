"""
Service layer: business logic that used to live directly inside API route
handlers.

Design decision (Service Layer Pattern): routers stay thin -- parse the
request, call one service method, shape the response. Every rule that isn't
purely an HTTP concern (batch size limits, risk-band bucketing, duplicate
username checks, promotion bookkeeping) lives here instead, expressed in
terms of repositories and domain exceptions rather than `Session` objects
and `HTTPException`. This is what makes the business logic testable without
FastAPI's `TestClient` and reusable from a future non-HTTP entrypoint (a CLI,
a Celery task, a gRPC service) without duplicating it.
"""

from __future__ import annotations

from services.auth_service import AuthService
from services.model_governance_service import ModelGovernanceService
from services.monitoring_service import MonitoringService
from services.scoring_service import ScoringService

__all__ = ["AuthService", "ModelGovernanceService", "MonitoringService", "ScoringService"]
