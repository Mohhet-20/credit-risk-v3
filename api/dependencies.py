"""
FastAPI dependency injection: DB sessions, repositories, services, current-user
resolution, RBAC, and a process-wide cached model bundle (so every request
doesn't re-load the model + feature pipeline from disk).

Design decision: this module is the ONLY place that wires concrete
repositories into concrete services. Routers depend on `get_auth_service`,
`get_scoring_service`, etc. -- never on a repository or a `Session` directly
-- so the composition root (which repository backs which service, which
cache-busting callback a service receives) lives in exactly one file. This
is the Dependency Injection pattern: services declare what they need in
their constructors; this module is where those needs get satisfied with
real objects for a live request (and, in tests, `app.dependency_overrides`
substitutes this whole chain with fixtures -- see `tests/conftest.py`).

Design decision: the champion model bundle is loaded ONCE at process start
(via `get_model_bundle`'s `lru_cache`) rather than per-request, since
deserializing a joblib model + feature pipeline is far too slow to repeat on
every call. `ModelGovernanceService.reload_cache()` (via `bust_model_cache`,
injected as a plain callable so the service layer doesn't import this API
module) explicitly busts this cache after a promotion/rollback, which is the
standard pattern for "hot-reloadable" model serving without restarting the
process.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from configs.settings import get_settings
from core.exceptions import AuthenticationError, AuthorizationError
from core.security import check_role, decode_token
from database.models import User
from database.session import get_db
from models.registry import ModelBundle, ModelRegistry
from repositories.audit_log_repository import AuditLogRepository
from repositories.business_rule_repository import BusinessRuleRepository
from repositories.data_quality_repository import DataQualityReportRepository
from repositories.model_version_repository import ModelVersionRepository
from repositories.prediction_repository import PredictionRepository
from repositories.system_config_repository import SystemConfigRepository
from repositories.user_repository import UserRepository
from repositories.workflow_execution_repository import WorkflowExecutionRepository
from services.audit_service import AuditService
from services.auth_service import AuthService
from services.config_service import ConfigService
from services.data_quality_service import DataQualityService
from services.governance_service import GovernanceService
from services.model_governance_service import ModelGovernanceService
from services.monitoring_service import MonitoringService
from services.notification_service import NotificationService
from services.rule_service import RuleService
from services.scoring_service import ScoringService
from services.user_service import UserService
from services.workflow_service import WorkflowService


# --- Model bundle ------------------------------------------------------------
@lru_cache
def _cached_bundle(version: str) -> ModelBundle:
    settings = get_settings()
    registry = ModelRegistry(settings.model_artifact_dir)
    return registry.load_version(version)


def get_model_registry() -> ModelRegistry:
    settings = get_settings()
    return ModelRegistry(settings.model_artifact_dir)


def get_model_bundle(registry: ModelRegistry = Depends(get_model_registry)) -> ModelBundle:
    version = registry.get_champion_version()
    if version is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No champion model has been promoted yet.")
    return _cached_bundle(version)


def bust_model_cache() -> None:
    _cached_bundle.cache_clear()


# --- Repositories ----------------------------------------------------------------
def get_user_repository(db: Session = Depends(get_db)) -> UserRepository:
    return UserRepository(db)


def get_prediction_repository(db: Session = Depends(get_db)) -> PredictionRepository:
    return PredictionRepository(db)


def get_audit_log_repository(db: Session = Depends(get_db)) -> AuditLogRepository:
    return AuditLogRepository(db)


def get_model_version_repository(db: Session = Depends(get_db)) -> ModelVersionRepository:
    return ModelVersionRepository(db)


def get_data_quality_repository(db: Session = Depends(get_db)) -> DataQualityReportRepository:
    return DataQualityReportRepository(db)


# --- Services --------------------------------------------------------------------
def get_auth_service(
    user_repo: UserRepository = Depends(get_user_repository),
    audit_repo: AuditLogRepository = Depends(get_audit_log_repository),
) -> AuthService:
    return AuthService(user_repo, audit_repo)


def get_scoring_service(
    prediction_repo: PredictionRepository = Depends(get_prediction_repository),
    audit_repo: AuditLogRepository = Depends(get_audit_log_repository),
) -> ScoringService:
    return ScoringService(prediction_repo, audit_repo)


def get_model_governance_service(
    registry: ModelRegistry = Depends(get_model_registry),
    model_version_repo: ModelVersionRepository = Depends(get_model_version_repository),
    audit_repo: AuditLogRepository = Depends(get_audit_log_repository),
) -> ModelGovernanceService:
    return ModelGovernanceService(registry, model_version_repo, audit_repo, bust_cache_fn=bust_model_cache)


def get_monitoring_service(
    prediction_repo: PredictionRepository = Depends(get_prediction_repository),
) -> MonitoringService:
    return MonitoringService(prediction_repo)


def get_data_quality_service(
    dq_repo: DataQualityReportRepository = Depends(get_data_quality_repository),
) -> DataQualityService:
    return DataQualityService(dq_repo)


def get_user_service(
    user_repo: UserRepository = Depends(get_user_repository),
    audit_repo: AuditLogRepository = Depends(get_audit_log_repository),
) -> UserService:
    return UserService(user_repo, audit_repo)


def get_audit_service(audit_repo: AuditLogRepository = Depends(get_audit_log_repository)) -> AuditService:
    return AuditService(audit_repo)


# --- Auth / RBAC -------------------------------------------------------------------
def get_current_user(
    authorization: str | None = Header(default=None),
    user_repo: UserRepository = Depends(get_user_repository),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or malformed Authorization header.")
    token = authorization.split(" ", 1)[1]
    try:
        payload = decode_token(token)
    except AuthenticationError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    if payload.get("type") != "access":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Provided token is not an access token.")

    user = user_repo.get_by_id(payload["sub"])
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive.")
    return user


def require_role(required_role: str):
    """Ordinal 'this tier or above' authorization -- see `core/security.py::RolePower`.
    Correct for every endpoint whose access model is a simple ladder
    (viewer < loan_officer < risk_analyst < admin < super_admin)."""

    def _dependency(current_user: User = Depends(get_current_user)) -> User:
        try:
            check_role(required_role, current_user.role)
        except AuthorizationError as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
        return current_user

    return _dependency


def require_roles(*allowed_roles: str):
    """Explicit role allow-list authorization, as opposed to `require_role()`'s
    ordinal semantics -- use this for endpoints that need to admit a LATERAL
    specialist role (`auditor`, `ml_engineer`) alongside an ordinal tier,
    without granting that specialist role everything the ordinal tier's
    superiors can do. Typical call site:

        Depends(require_roles(*roles_at_least("risk_analyst"), "auditor"))

    which reads as "risk_analyst or above, OR an auditor" -- `roles_at_least()`
    (in `core/security.py`) expands the ordinal part of that list so call
    sites never hand-enumerate ordinal role names."""
    allowed = set(allowed_roles)

    def _dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Role '{current_user.role}' is not permitted for this action.",
            )
        return current_user

    return _dependency


# --- Business Rule Engine / Workflow Engine / Configuration Center ---------------
def get_business_rule_repository(db: Session = Depends(get_db)) -> BusinessRuleRepository:
    return BusinessRuleRepository(db)


def get_workflow_execution_repository(db: Session = Depends(get_db)) -> WorkflowExecutionRepository:
    return WorkflowExecutionRepository(db)


def get_system_config_repository(db: Session = Depends(get_db)) -> SystemConfigRepository:
    return SystemConfigRepository(db)


def get_rule_service(
    rule_repo: BusinessRuleRepository = Depends(get_business_rule_repository),
    audit_repo: AuditLogRepository = Depends(get_audit_log_repository),
) -> RuleService:
    return RuleService(rule_repo, audit_repo)


def get_config_service(
    config_repo: SystemConfigRepository = Depends(get_system_config_repository),
    audit_repo: AuditLogRepository = Depends(get_audit_log_repository),
) -> ConfigService:
    return ConfigService(config_repo, audit_repo)


def get_notification_service(
    audit_repo: AuditLogRepository = Depends(get_audit_log_repository),
    config_service: ConfigService = Depends(get_config_service),
) -> NotificationService:
    return NotificationService(audit_repo, config_service)


def get_workflow_service(
    workflow_repo: WorkflowExecutionRepository = Depends(get_workflow_execution_repository),
    scoring_service: ScoringService = Depends(get_scoring_service),
    rule_service: RuleService = Depends(get_rule_service),
    config_service: ConfigService = Depends(get_config_service),
    notification_service: NotificationService = Depends(get_notification_service),
    audit_repo: AuditLogRepository = Depends(get_audit_log_repository),
) -> WorkflowService:
    return WorkflowService(
        workflow_repo,
        scoring_service,
        rule_service,
        config_service,
        notification_service,
        audit_repo,
    )


def get_governance_service(
    governance: ModelGovernanceService = Depends(get_model_governance_service),
    rule_service: RuleService = Depends(get_rule_service),
    workflow_service: WorkflowService = Depends(get_workflow_service),
    dq_service: DataQualityService = Depends(get_data_quality_service),
) -> GovernanceService:
    return GovernanceService(governance, rule_service, workflow_service, dq_service)
