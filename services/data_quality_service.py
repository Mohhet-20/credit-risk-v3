"""
Data-quality orchestration: runs validation + profiling + scoring against a
dataset and persists the combined result via `DataQualityReportRepository`.

Design decision: `DataQualityReportRepository` (and the `DataQualityReport`
table it backs) has existed since the earlier Repository + Service layer
pass, but nothing ever called `save_report` -- validation/profiling results
were computed on demand (by the training pipeline, by ad hoc scripts) and
thrown away, never queryable as history. This service closes that gap,
mirroring the same "the table and repository exist but nobody reads/writes
through them" pattern already fixed for the audit log in an earlier pass.
"""

from __future__ import annotations

import pandas as pd

from core.pagination import Page, PaginationParams
from data_quality.profiling import generate_profile
from data_quality.scoring import compute_quality_score
from data_quality.validators import BusinessRule, run_full_validation
from repositories.data_quality_repository import DataQualityReportRepository

REPORT_TYPE = "quality_report"


class DataQualityService:
    def __init__(self, repo: DataQualityReportRepository):
        self.repo = repo

    def run_and_persist(
        self,
        df: pd.DataFrame,
        *,
        target: pd.Series | None = None,
        outlier_columns: list[str] | None = None,
        duplicate_subset: list[str] | None = None,
        business_rules: list[BusinessRule] | None = None,
        dataset_path: str | None = None,
        dataset_label: str = "reference",
    ) -> dict:
        validation_report = run_full_validation(
            df,
            target=target,
            outlier_columns=outlier_columns,
            duplicate_subset=duplicate_subset,
            business_rules=business_rules,
        )
        profile = generate_profile(df, target=target)
        score = compute_quality_score(validation_report, profile, dataset_path=dataset_path)

        summary = {
            "dataset_label": dataset_label,
            "validation": validation_report.summary(),
            "profile": profile,
            "score": score,
        }
        record = self.repo.save_report(report_type=REPORT_TYPE, summary=summary, passed=validation_report.passed)
        return self._to_dict(record)

    def get_latest(self) -> dict | None:
        reports = self.repo.list_reports(report_type=REPORT_TYPE, limit=1)
        return self._to_dict(reports[0]) if reports else None

    def list_history(self, params: PaginationParams) -> Page[dict]:
        offset, limit = params.to_offset_limit()
        items, total = self.repo.list_reports_page(report_type=REPORT_TYPE, limit=limit, offset=offset)
        return Page.create(items=[self._to_dict(r) for r in items], total=total, params=params)

    @staticmethod
    def _to_dict(record) -> dict:
        return {
            "id": record.id,
            "report_type": record.report_type,
            "passed": record.passed,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            **record.summary,
        }
