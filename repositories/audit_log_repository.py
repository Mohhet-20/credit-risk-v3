"""Audit log aggregate repository."""

from __future__ import annotations

from sqlalchemy import func, select

from database.models import AuditLog
from repositories.base import BaseRepository


class AuditLogRepository(BaseRepository[AuditLog]):
    model = AuditLog

    def log_action(
        self,
        *,
        user_id: str | None,
        action: str,
        resource: str | None = None,
        details: dict | None = None,
        ip_address: str | None = None,
    ) -> AuditLog:
        entry = AuditLog(user_id=user_id, action=action, resource=resource, details=details, ip_address=ip_address)
        return self.add(entry)

    def list_audit_logs(self, limit: int = 200, offset: int = 0) -> list[AuditLog]:
        stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)
        return list(self.db.scalars(stmt))

    def list_audit_logs_page(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        sort_order: str = "desc",
        user_id: str | None = None,
        action: str | None = None,
    ) -> tuple[list[AuditLog], int]:
        """Filtered + paginated listing for the admin audit-log endpoint.
        Returns `(items, total_count)` -- see `UserRepository.list_users_page`
        for the same pattern."""
        stmt = select(AuditLog)
        count_stmt = select(func.count()).select_from(AuditLog)
        if user_id:
            stmt = stmt.where(AuditLog.user_id == user_id)
            count_stmt = count_stmt.where(AuditLog.user_id == user_id)
        if action:
            stmt = stmt.where(AuditLog.action == action)
            count_stmt = count_stmt.where(AuditLog.action == action)

        order = AuditLog.created_at.asc() if sort_order == "asc" else AuditLog.created_at.desc()
        stmt = stmt.order_by(order).limit(limit).offset(offset)

        items = list(self.db.scalars(stmt))
        total = self.db.scalar(count_stmt) or 0
        return items, total
