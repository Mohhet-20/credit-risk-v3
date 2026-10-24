"""
Enterprise Configuration Center endpoints: admin-editable business-rule
thresholds, feature flags, notification settings, and workflow toggles --
see `services/config_service.py` for the single-table `SystemConfig`
design decision and `DEFAULT_CONFIG` fallback semantics this router
exposes.

RBAC: reading configuration is `risk_analyst`-or-above (operators need to
see current thresholds, and this additionally admits `auditor`/
`ml_engineer` per the usual lateral-role pattern); writing configuration
is `admin`-or-above only -- these keys directly control automated
decisioning behavior (e.g. `workflow.manual_review_band`), so changing
them is treated the same as any other admin-only platform-configuration
action.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_config_service, require_role, require_roles
from api.schemas import SystemConfigOut, SystemConfigSetRequest
from core.security import roles_at_least
from database.models import User
from services.config_service import ConfigService

router = APIRouter(prefix="/config", tags=["config"])

_READ_ROLES = (*roles_at_least("risk_analyst"), "auditor", "ml_engineer")


@router.get("", response_model=list[SystemConfigOut])
def list_config(
    category: str | None = None,
    current_user: User = Depends(require_roles(*_READ_ROLES)),
    service: ConfigService = Depends(get_config_service),
):
    return service.list_all(category=category)


@router.get("/{key}")
def get_config_value(
    key: str,
    current_user: User = Depends(require_roles(*_READ_ROLES)),
    service: ConfigService = Depends(get_config_service),
):
    return {"key": key, "value": service.get(key)}


@router.put("/{key}", response_model=SystemConfigOut)
def set_config_value(
    key: str,
    payload: SystemConfigSetRequest,
    current_user: User = Depends(require_role("admin")),
    service: ConfigService = Depends(get_config_service),
):
    return service.set(
        key,
        payload.value,
        category=payload.category,
        description=payload.description,
        actor_user_id=current_user.id,
    )
