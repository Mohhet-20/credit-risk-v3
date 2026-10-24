"""
Audit-log query business logic.

Design decision: this service exists purely to expose the `AuditLog` table
(already written to by every other service via `AuditLogRepository.log_action`)
through a real, paginated, filterable API -- before this, audit entries were
persisted on every login/predict/promote/rollback but nothing ever read them
back out. That's a genuine gap for an "enterprise" platform: audit data that
can't be queried isn't actually an audit trail.
"""

from __future__ import annotations

from core.pagination import Page, PaginationParams
from database.models import AuditLog
from repositories.audit_log_repository import AuditLogRepository


class AuditService:
    def __init__(self, audit_repo: AuditLogRepository):
        self.audit_repo = audit_repo

    def list_logs(
        self,
        params: PaginationParams,
        *,
        user_id: str | None = None,
        action: str | None = None,
    ) -> Page[AuditLog]:
        offset, limit = params.to_offset_limit()
        items, total = self.audit_repo.list_audit_logs_page(
            limit=limit,
            offset=offset,
            sort_order=params.sort_order,
            user_id=user_id,
            action=action,
        )
        return Page.create(items=items, total=total, params=params)
