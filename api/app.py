"""
FastAPI application entrypoint.

Design decision: routers are versioned under `settings.api_v1_prefix`
(`/api/v1` by default) rather than unversioned paths, so a future `/api/v2`
can be introduced without breaking existing API consumers -- standard
practice for any API meant to be depended on by more than one internal
client (the Streamlit frontend today, external systems tomorrow). Rate
limiting (slowapi / token-bucket per client IP) is applied globally via
middleware rather than per-route, so every endpoint is protected by default
and routes only need to opt OUT (rare) rather than remembering to opt in;
`/auth/login` and `/predict/batch` additionally apply a stricter
route-level limit (see `core/rate_limit.py`) since they're higher-value
abuse targets than the average endpoint.

Design decision: exception handlers are registered from MOST to LEAST
specific subclass of `CreditRiskPlatformError` (Starlette resolves a raised
exception's handler by walking its MRO and taking the first registered
match). `DuplicateResourceError`, `ModelNotFoundError`, and
`ResourceNotFoundError` need their own handlers (409 / 404 / 404) precisely
because they're domain exceptions raised by the service layer (`services/`)
rather than `HTTPException`s raised directly in a route -- see
`services/auth_service.py` and `services/model_governance_service.py`.
`AccountLockedError` needs no handler of its own: it's a *subclass* of
the existing `AuthenticationError`, so the existing 401 handler already
covers it via MRO. Every OTHER `CreditRiskPlatformError` subclass without a
dedicated handler still gets a sane default via the generic handler at the
bottom, so adding a new domain exception never produces an unhandled 500 by
omission. `InvalidRuleTypeError`/`InvalidRuleConditionError` (the
Business Rule Engine) get their own handlers too, despite being `ValueError`
subclasses rather than `CreditRiskPlatformError` subclasses -- see the
comment above their handlers below for why.

Design decision: two new middleware classes (`CorrelationIdMiddleware`,
`SecurityHeadersMiddleware`, in `api/middleware.py`) are registered LAST so
they end up OUTERMOST in the middleware stack -- Starlette builds the stack
in reverse registration order. `CorrelationIdMiddleware` must wrap
everything else so a correlation ID exists before CORS/rate-limit
middleware (or anything else) logs a line; `SecurityHeadersMiddleware`
runs last on the way out so its headers are never overwritten downstream.

Reliability (this pass): startup/shutdown moved from the deprecated
`@app.on_event("startup")` decorator to a `lifespan` async context manager
(FastAPI's documented replacement -- `on_event` is deprecated upstream and
was the source of a `DeprecationWarning` on every test run). The lifespan
now does three things on startup, in order: (1) `settings.validate_or_raise()`
-- fail fast if this is a production deployment with an insecure default
JWT secret, debug mode on, or SQLite as the database (see
`configs/settings.py`); (2) `init_db_with_retry()` instead of a bare
`init_db()`, so a `docker compose up` cold start racing Postgres's startup
doesn't crash-loop the API container; (3) a log line marking the app ready.
On shutdown it releases the cache backend's connection (`get_cache().close()`,
a no-op for the in-memory backend, a real connection-pool release for
Redis) and disposes the SQLAlchemy engine's connection pool
(`engine.dispose()`) so no sockets are left open across a graceful restart.

Run locally with: `uvicorn api.app:app --reload`
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

import core.sklearn_compat  # noqa: F401  (side-effect: patches XGBClassifier sklearn tags)
from api.dependencies import get_model_registry
from api.middleware import CorrelationIdMiddleware, SecurityHeadersMiddleware
from api.routers import (
    admin,
    analytics,
    auth,
    copilot,
    data_quality,
    explain,
    models,
    monitoring,
    reports,
    scoring,
)
from api.routers import (
    config as config_router,
)
from api.routers import health as health_router
from api.routers import (
    rules as rules_router,
)
from api.routers import (
    workflow as workflow_router,
)
from configs.settings import get_settings
from core.cache import get_cache
from core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    CreditRiskPlatformError,
    DuplicateResourceError,
    ModelNotFoundError,
    ResourceNotFoundError,
)
from core.logging import configure_logging, get_correlation_id, get_logger
from core.rate_limit import limiter
from database.session import engine, init_db_with_retry
from rules.engine import InvalidRuleConditionError
from services.rule_service import InvalidRuleTypeError

settings = get_settings()
configure_logging(level=settings.log_level)
logger = get_logger(__name__)

TAGS_METADATA = [
    {"name": "auth", "description": "Registration, login, and token refresh."},
    {"name": "prediction", "description": "Single, batch, and async credit-risk scoring."},
    {"name": "explainability", "description": "SHAP/LIME explanations and counterfactual scenarios."},
    {"name": "models", "description": "Model registry: list versions, compare, promote, rollback."},
    {"name": "monitoring", "description": "Prediction distribution, latency, and drift monitoring."},
    {"name": "analytics", "description": "Segmentation, anomaly detection, and loan recommendations."},
    {"name": "admin", "description": "Admin-only user management and audit-log queries."},
    {"name": "health", "description": "Liveness/readiness probes and diagnostics for operators."},
    {
        "name": "reports",
        "description": "Downloadable HTML reports: comparison, evaluation, drift, "
        "performance, data quality, explainability.",
    },
    {
        "name": "data-quality",
        "description": "Run/read validation + profiling + composite quality-score "
        "checks against the reference dataset, with history.",
    },
    {
        "name": "copilot",
        "description": "AI Copilot: explain predictions, summarize model/monitoring/data-quality reports, and generate an executive briefing (deterministic template-based NLG, no external LLM dependency).",
    },
    {
        "name": "rules",
        "description": "Business Rule Engine: admin-editable approval/rejection/manual-review rules "
        "(closed JSON condition grammar, versioned, audit-logged) plus dry-run simulation.",
    },
    {
        "name": "workflow",
        "description": "Enterprise Workflow Engine: the full governed credit-decisioning pipeline "
        "(validation, data quality, fraud check, AI prediction, business rules, "
        "manual review, decision, notification, audit log), additive alongside "
        "the direct /predict endpoint.",
    },
    {
        "name": "config",
        "description": "Enterprise Configuration Center: feature flags, thresholds, and workflow "
        "toggles, centrally managed instead of hard-coded in application code.",
    },
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.validate_or_raise()
    init_db_with_retry()
    logger.info("Database initialized", extra={"context": {"environment": settings.app_env}})
    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown starting")
    try:
        get_cache().close()
    except Exception as exc:  # noqa: BLE001 -- shutdown must not hang/crash on a cache-close failure
        logger.warning("Error closing cache backend during shutdown", extra={"context": {"error": str(exc)}})
    engine.dispose()
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    description=(
        "Enterprise Credit Risk Intelligence Platform API -- model-backed credit "
        "risk scoring, explainability, monitoring, and model governance."
    ),
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CorrelationIdMiddleware)


@app.middleware("http")
async def request_timing_and_logging(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code}",
        extra={"context": {"duration_ms": round(duration_ms, 2), "correlation_id": get_correlation_id()}},
    )
    response.headers["X-Process-Time-Ms"] = str(round(duration_ms, 2))
    return response


@app.exception_handler(AuthenticationError)
async def auth_error_handler(request: Request, exc: AuthenticationError):
    return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": str(exc)})


@app.exception_handler(AuthorizationError)
async def authz_error_handler(request: Request, exc: AuthorizationError):
    return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": str(exc)})


@app.exception_handler(DuplicateResourceError)
async def duplicate_resource_handler(request: Request, exc: DuplicateResourceError):
    return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})


@app.exception_handler(ResourceNotFoundError)
async def resource_not_found_handler(request: Request, exc: ResourceNotFoundError):
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})


@app.exception_handler(CreditRiskPlatformError)
async def platform_error_handler(request: Request, exc: CreditRiskPlatformError):
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


# `InvalidRuleTypeError` and `InvalidRuleConditionError` (the Business
# Rule Engine) are intentionally `ValueError` subclasses, not
# `CreditRiskPlatformError` subclasses -- `rules/engine.py` is a pure,
# framework-agnostic module (see its docstring) that shouldn't import an
# API-layer exception hierarchy just to raise a validation error. This is
# the API layer's job: translate them into the same 400 response shape
# every other malformed-request error gets.
@app.exception_handler(InvalidRuleTypeError)
async def invalid_rule_type_handler(request: Request, exc: InvalidRuleTypeError):
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


@app.exception_handler(InvalidRuleConditionError)
async def invalid_rule_condition_handler(request: Request, exc: InvalidRuleConditionError):
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


@app.get("/health", tags=["health"])
def health():
    registry = get_model_registry()
    champion_version = registry.get_champion_version()
    champion_name = None
    if champion_version:
        try:
            champion_name = registry.load_version(champion_version).metadata.get("model_name")
        except Exception:  # noqa: BLE001
            champion_name = None
    return {"status": "ok", "champion_model": champion_name, "champion_version": champion_version}


API_PREFIX = settings.api_v1_prefix
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(scoring.router, prefix=API_PREFIX)
app.include_router(explain.router, prefix=API_PREFIX)
app.include_router(models.router, prefix=API_PREFIX)
app.include_router(monitoring.router, prefix=API_PREFIX)
app.include_router(analytics.router, prefix=API_PREFIX)
app.include_router(admin.router, prefix=API_PREFIX)
app.include_router(copilot.router, prefix=API_PREFIX)
app.include_router(reports.router, prefix=API_PREFIX)
app.include_router(data_quality.router, prefix=API_PREFIX)
app.include_router(rules_router.router, prefix=API_PREFIX)
app.include_router(workflow_router.router, prefix=API_PREFIX)
app.include_router(config_router.router, prefix=API_PREFIX)

# Mounted at root (not under API_PREFIX): health/diagnostic endpoints are an
# operational surface for load balancers, Kubernetes probes, and uptime
# monitors, not a versioned business API -- an infra tool checking
# `/health/liveness` shouldn't need to track API version bumps.
app.include_router(health_router.router)
