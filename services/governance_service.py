"""
AI Governance composition service: aggregates model-governance status,
business-rule inventory, recent workflow outcomes, and data-quality status
into a single "compliance dashboard" snapshot -- the numbers a compliance/
audit review would otherwise have to pull from four separate endpoints.

Design decision: this service computes NOTHING new. Every field it returns
is read straight from `ModelGovernanceService`, `RuleService`,
`WorkflowService`, and `DataQualityService`, which already track this data
for their own endpoints. Factored out here (rather than duplicated inline
in both `api/routers/reports.py`'s `/compliance-dashboard` and
`api/routers/copilot.py`'s `/summary/governance`) so the two call sites can
never silently drift into reporting different numbers for the same
underlying state.
"""

from __future__ import annotations

from services.data_quality_service import DataQualityService
from services.model_governance_service import ModelGovernanceService
from services.rule_service import RuleService
from services.workflow_service import WorkflowService


class GovernanceService:
    def __init__(
        self,
        governance: ModelGovernanceService,
        rule_service: RuleService,
        workflow_service: WorkflowService,
        dq_service: DataQualityService,
    ):
        self.governance = governance
        self.rule_service = rule_service
        self.workflow_service = workflow_service
        self.dq_service = dq_service

    def compliance_dashboard(self, *, workflow_window_days: int = 30) -> dict:
        versions = self.governance.list_versions()
        champion = next((v for v in versions if v.get("is_champion")), None)

        all_rules = self.rule_service.list_rules()
        active_rules = [r for r in all_rules if r.is_active]
        rules_by_type: dict[str, int] = {}
        for r in active_rules:
            rules_by_type[r.rule_type] = rules_by_type.get(r.rule_type, 0) + 1

        workflow_stats = self.workflow_service.stats(days=workflow_window_days)
        dq_result = self.dq_service.get_latest()

        return {
            "model_governance": {
                "champion_version": champion["version"] if champion else None,
                "champion_model_name": champion.get("model_name") if champion else None,
                "total_versions": len(versions),
            },
            "business_rules": {
                "total_rules": len(all_rules),
                "active_rules": len(active_rules),
                "active_rules_by_type": rules_by_type,
            },
            "workflow": workflow_stats,
            "data_quality": {
                "latest_report_passed": dq_result.get("passed") if dq_result else None,
                "latest_overall_score": dq_result.get("score", {}).get("overall_score") if dq_result else None,
                "has_report": dq_result is not None,
            },
        }
