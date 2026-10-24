"""Data quality report aggregate repository."""

from __future__ import annotations

from sqlalchemy import func, select

from database.models import DataQualityReport
from repositories.base import BaseRepository


class DataQualityReportRepository(BaseRepository[DataQualityReport]):
    model = DataQualityReport

    def save_report(self, *, report_type: str, summary: dict, passed: bool) -> DataQualityReport:
        report = DataQualityReport(report_type=report_type, summary=summary, passed=passed)
        return self.add(report)

    def list_reports(
        self,
        report_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DataQualityReport]:
        stmt = select(DataQualityReport)
        if report_type:
            stmt = stmt.where(DataQualityReport.report_type == report_type)
        stmt = stmt.order_by(DataQualityReport.created_at.desc()).limit(limit).offset(offset)
        return list(self.db.scalars(stmt))

    def list_reports_page(
        self,
        *,
        report_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[DataQualityReport], int]:
        """Paginated listing (see `UserRepository.list_users_page` for the
        same `(items, total)` pattern used across every other admin/history
        listing endpoint in this codebase)."""
        stmt = select(DataQualityReport)
        count_stmt = select(func.count()).select_from(DataQualityReport)
        if report_type:
            stmt = stmt.where(DataQualityReport.report_type == report_type)
            count_stmt = count_stmt.where(DataQualityReport.report_type == report_type)
        stmt = stmt.order_by(DataQualityReport.created_at.desc()).limit(limit).offset(offset)

        items = list(self.db.scalars(stmt))
        total = self.db.scalar(count_stmt) or 0
        return items, total
