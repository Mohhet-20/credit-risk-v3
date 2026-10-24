"""Repository for `WorkflowExecution` rows -- see `workflows/engine.py`
for the orchestration logic that writes these."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import case, func, select

from database.models import WorkflowExecution
from repositories.base import BaseRepository


class WorkflowExecutionRepository(BaseRepository[WorkflowExecution]):
    model = WorkflowExecution

    def list_recent(self, limit: int = 100) -> list[WorkflowExecution]:
        stmt = select(WorkflowExecution).order_by(WorkflowExecution.started_at.desc()).limit(limit)
        return list(self.db.scalars(stmt))

    def stats_last_n_days(self, days: int = 30) -> dict:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        stmt = select(
            func.count(WorkflowExecution.id),
            func.sum(case((WorkflowExecution.status == "completed", 1), else_=0)),
            func.sum(case((WorkflowExecution.status == "failed", 1), else_=0)),
        ).where(WorkflowExecution.started_at >= since)
        total, completed, failed = self.db.execute(stmt).one()
        return {
            "total": int(total or 0),
            "completed": int(completed or 0),
            "failed": int(failed or 0),
            "window_days": days,
        }
