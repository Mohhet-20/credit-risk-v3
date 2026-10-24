"""
Admin user-management business logic: list/get/activate/deactivate/change
role/unlock, split out of `AuthService`.

Design decision: `AuthService` owns the *self-service* identity lifecycle
(register, login, refresh) while `UserService` owns *admin-on-someone-else*
operations (list all users, deactivate a user, change a user's role, clear a
lockout). These are genuinely different responsibilities with different
callers and different RBAC requirements (`UserService` methods are only ever
reached from admin-gated routes) -- splitting them keeps each service small
and keeps `AuthService`'s unit tests free of admin-only edge cases.
"""

from __future__ import annotations

from core.constants import UserRole
from core.exceptions import AuthorizationError, ResourceNotFoundError
from core.pagination import Page, PaginationParams
from database.models import User
from repositories.audit_log_repository import AuditLogRepository
from repositories.user_repository import UserRepository


class UserService:
    def __init__(self, user_repo: UserRepository, audit_repo: AuditLogRepository):
        self.user_repo = user_repo
        self.audit_repo = audit_repo

    def list_users(self, params: PaginationParams) -> Page[User]:
        offset, limit = params.to_offset_limit()
        items, total = self.user_repo.list_users_page(
            limit=limit,
            offset=offset,
            sort_by=params.sort_by,
            sort_order=params.sort_order,
        )
        return Page.create(items=items, total=total, params=params)

    def get_user(self, user_id: str) -> User:
        user = self.user_repo.get_by_id(user_id)
        if user is None:
            raise ResourceNotFoundError(f"User '{user_id}' not found.")
        return user

    def set_active_status(self, user_id: str, *, is_active: bool, actor_user_id: str) -> User:
        if user_id == actor_user_id and not is_active:
            raise AuthorizationError("You cannot deactivate your own account.")
        user = self.get_user(user_id)
        user = self.user_repo.set_active_status(user, is_active=is_active)
        self.audit_repo.log_action(
            user_id=actor_user_id,
            action="user_deactivated" if not is_active else "user_activated",
            resource=f"user:{user_id}",
        )
        return user

    # Roles that can grant `/admin/*` access -- an admin-tier user changing
    # their OWN role away from one of these would either lock every admin
    # out of user management (if they were the last one) or at minimum
    # requires a second admin to undo, so it's blocked the same way
    # self-deactivation is blocked above.
    _ADMIN_TIER_ROLES = frozenset({"admin", "super_admin"})

    def set_role(self, user_id: str, *, role: str, actor_user_id: str) -> User:
        if role not in {r.value for r in UserRole}:
            raise ResourceNotFoundError(f"Unknown role '{role}'.")
        user = self.get_user(user_id)
        if user_id == actor_user_id and user.role in self._ADMIN_TIER_ROLES and role not in self._ADMIN_TIER_ROLES:
            raise AuthorizationError("You cannot remove your own admin-tier role.")
        user = self.user_repo.set_role(user, role=role)
        self.audit_repo.log_action(
            user_id=actor_user_id,
            action="user_role_changed",
            resource=f"user:{user_id}",
            details={"role": role},
        )
        return user

    def unlock_account(self, user_id: str, *, actor_user_id: str) -> User:
        user = self.get_user(user_id)
        user = self.user_repo.unlock_account(user)
        self.audit_repo.log_action(user_id=actor_user_id, action="user_unlocked", resource=f"user:{user_id}")
        return user
