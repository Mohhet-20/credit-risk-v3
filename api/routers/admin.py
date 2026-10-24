"""
Admin endpoints: paginated user management, privileged user creation, and
audit-log queries.

Design decision: user-management routes (list/get/status/role/unlock/create)
are each gated with `require_role("admin")` individually rather than a
blanket router-level dependency -- `GET /admin/audit-logs` needs to ALSO
admit the lateral `auditor` role (via `require_roles`), which a router-level
"admin only" dependency would make impossible to loosen per-route (FastAPI
dependencies compose as AND, not override). Every user-management route's
actual access (admin/super_admin only) is unchanged from before this split;
only audit-log reading gained the additional `auditor` grant.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_audit_service, get_auth_service, get_user_service, require_role, require_roles
from api.schemas import (
    AdminUserCreate,
    AdminUserOut,
    AdminUserPage,
    AuditLogPage,
    UpdateUserRoleRequest,
    UpdateUserStatusRequest,
)
from core.pagination import PaginationParams, pagination_params
from core.security import roles_at_least
from database.models import User
from services.audit_service import AuditService
from services.auth_service import AuthService
from services.user_service import UserService

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=AdminUserPage, dependencies=[Depends(require_role("admin"))])
def list_users(
    params: PaginationParams = Depends(pagination_params),
    user_service: UserService = Depends(get_user_service),
):
    return user_service.list_users(params)


@router.post("/users", response_model=AdminUserOut, status_code=201)
def create_user(
    payload: AdminUserCreate,
    current_user: User = Depends(require_role("admin")),
    auth_service: AuthService = Depends(get_auth_service),
):
    """The only place a privileged role (super_admin/admin/ml_engineer/
    risk_analyst/auditor) can be assigned at account-creation time --
    `POST /auth/register` (public, unauthenticated) is restricted to
    self-service roles only. See `services/auth_service.py::AuthService.register`."""
    return auth_service.register(
        username=payload.username,
        email=payload.email,
        password=payload.password,
        role=payload.role,
        requesting_user=current_user,
    )


@router.get("/users/{user_id}", response_model=AdminUserOut, dependencies=[Depends(require_role("admin"))])
def get_user(user_id: str, user_service: UserService = Depends(get_user_service)):
    return user_service.get_user(user_id)


@router.patch("/users/{user_id}/status", response_model=AdminUserOut)
def update_user_status(
    user_id: str,
    payload: UpdateUserStatusRequest,
    current_user: User = Depends(require_role("admin")),
    user_service: UserService = Depends(get_user_service),
):
    return user_service.set_active_status(user_id, is_active=payload.is_active, actor_user_id=current_user.id)


@router.patch("/users/{user_id}/role", response_model=AdminUserOut)
def update_user_role(
    user_id: str,
    payload: UpdateUserRoleRequest,
    current_user: User = Depends(require_role("admin")),
    user_service: UserService = Depends(get_user_service),
):
    return user_service.set_role(user_id, role=payload.role, actor_user_id=current_user.id)


@router.post("/users/{user_id}/unlock", response_model=AdminUserOut)
def unlock_user(
    user_id: str,
    current_user: User = Depends(require_role("admin")),
    user_service: UserService = Depends(get_user_service),
):
    return user_service.unlock_account(user_id, actor_user_id=current_user.id)


@router.get(
    "/audit-logs",
    response_model=AuditLogPage,
    dependencies=[Depends(require_roles(*roles_at_least("admin"), "auditor"))],
)
def list_audit_logs(
    user_id: str | None = None,
    action: str | None = None,
    params: PaginationParams = Depends(pagination_params),
    audit_service: AuditService = Depends(get_audit_service),
):
    return audit_service.list_logs(params, user_id=user_id, action=action)
