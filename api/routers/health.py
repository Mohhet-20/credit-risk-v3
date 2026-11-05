"""
Health and diagnostics endpoints for orchestrators (Kubernetes liveness/
readiness probes, load balancer health checks) and operators (version,
database, system).

Design decision: this is deliberately a NEW router mounted at root
(`/health/...`), additive alongside the existing bare `GET /health` in
`api/app.py`. The existing endpoint is already referenced by
`deployment/kubernetes/*.yaml` and both Dockerfiles' `HEALTHCHECK`
directives, verified working in earlier work -- changing its shape or
path would risk breaking already-deployed infrastructure for zero benefit.
`/health/liveness` and `/health/readiness` express the standard Kubernetes
distinction the single bare endpoint never could: liveness answers "is the
process alive" (always 200 if this code is running at all) while readiness
answers "can this process serve traffic right now" (503 if the DB or model
isn't available), which controls two different Kubernetes behaviors
(restart vs. remove-from-service).

Design decision: the model registry is obtained via `Depends(get_model_registry)`
in every handler that needs it, rather than calling `get_model_registry()`
directly -- calling it directly would bypass `app.dependency_overrides`, so
`tests/conftest.py`'s fixture-model override would never apply here and
these routes would try to hit a real (nonexistent, in tests) model registry
on disk regardless of what the rest of the test suite has stubbed.
"""

from __future__ import annotations

import platform
import sys
import time

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text

from api.dependencies import get_model_registry
from configs.settings import get_settings
from database.session import SessionLocal
from models.registry import ModelRegistry

router = APIRouter(prefix="/health", tags=["health"])

_PROCESS_STARTED_AT = time.monotonic()


@router.get("/liveness")
def liveness():
    """Always 200 if the ASGI app can handle a request at all -- no I/O, no
    dependencies. A failure here means the process itself is wedged and
    Kubernetes should restart the pod."""
    return {"status": "alive"}


@router.get("/readiness")
def readiness(response: Response, registry: ModelRegistry = Depends(get_model_registry)):
    """503 if the database is unreachable or no champion model is loaded --
    signals Kubernetes to pull this pod out of the service's endpoint list
    without restarting it (the process itself may be fine; a dependency
    isn't)."""
    checks = {"database": _check_database(), "model": _check_model(registry)}
    healthy = all(checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if healthy else "not_ready", "checks": checks}


@router.get("/version")
def version():
    settings = get_settings()
    return {
        "app_name": settings.app_name,
        "version": "2.2.0",
        "environment": settings.app_env,
        "python_version": sys.version.split()[0],
    }


@router.get("/database")
def database_health(response: Response):
    ok = _check_database()
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    settings = get_settings()
    return {"status": "ok" if ok else "unreachable", "dialect": settings.database_url.split("://")[0]}


@router.get("/model")
def model_health(response: Response, registry: ModelRegistry = Depends(get_model_registry)):
    ok = _check_model(registry)
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if ok else "no_champion", "champion_version": registry.get_champion_version()}


@router.get("/system")
def system_health():
    return {
        "uptime_seconds": round(time.monotonic() - _PROCESS_STARTED_AT, 2),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _check_database() -> bool:
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            return True
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        return False


def _check_model(registry: ModelRegistry) -> bool:
    try:
        return registry.get_champion_version() is not None
    except Exception:  # noqa: BLE001
        return False
