"""
Enterprise Workflow Engine service: orchestrates the standard credit-risk
pipeline (see `workflows/engine.py`'s module docstring for the full step
list) by calling existing, already-tested services in sequence, recording
per-step timing/outcome via `WorkflowRunner`, and persisting the whole run
as a `WorkflowExecution` row.

Design decision: this is ADDITIVE alongside the existing `POST /predict`
endpoint (`ScoringService.score_single`), not a replacement for it.
`POST /predict` stays the fast, minimal-overhead scoring path (validate ->
predict -> persist -> return) that existing integrations depend on;
`POST /workflow/execute` is the fuller, governed pipeline (adds explicit
per-row validation, business-rule evaluation, a manual-review gate, and a
recorded execution trail) for callers that want the complete, auditable
enterprise flow. Both ultimately call the same `ScoringService`/model
bundle underneath, so a prediction produced either way is scored
identically -- the workflow engine adds process around the prediction, it
doesn't compute a different one.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from api.schemas import CreditApplicationRequest
from core.exceptions import DataValidationError, ResourceNotFoundError
from data_quality.validators import default_credit_business_rules, validate_business_rules, validate_schema
from database.models import WorkflowExecution
from models.registry import ModelBundle
from repositories.audit_log_repository import AuditLogRepository
from repositories.workflow_execution_repository import WorkflowExecutionRepository
from services.config_service import ConfigService
from services.notification_service import NotificationService
from services.rule_service import RuleService
from services.scoring_service import ScoringService
from workflows.engine import WorkflowRunner


class WorkflowService:
    def __init__(
        self,
        workflow_repo: WorkflowExecutionRepository,
        scoring_service: ScoringService,
        rule_service: RuleService,
        config_service: ConfigService,
        notification_service: NotificationService,
        audit_repo: AuditLogRepository,
    ):
        self.workflow_repo = workflow_repo
        self.scoring_service = scoring_service
        self.rule_service = rule_service
        self.config_service = config_service
        self.notification_service = notification_service
        self.audit_repo = audit_repo

    def list_executions(self, *, limit: int = 100) -> list[WorkflowExecution]:
        """Most recent workflow executions -- `GET /workflow/executions`.
        Kept as a thin pass-through so the router never touches
        `self.workflow_repo` directly (see the layered-architecture
        convention in `api/routers/`'s module docstrings)."""
        return self.workflow_repo.list_recent(limit=limit)

    def get_execution(self, execution_id: str) -> WorkflowExecution:
        execution = self.workflow_repo.get_by_id(execution_id)
        if execution is None:
            raise ResourceNotFoundError(f"Workflow execution '{execution_id}' not found.")
        return execution

    def stats(self, *, days: int = 30) -> dict:
        return self.workflow_repo.stats_last_n_days(days=days)

    def execute(
        self,
        application: CreditApplicationRequest,
        bundle: ModelBundle,
        *,
        requested_by: str | None,
    ) -> dict:
        runner = WorkflowRunner()
        prediction_result: dict | None = None
        rule_result: dict | None = None
        needs_manual_review = False
        final_decision: str | None = None

        # 1. Validation -- schema conformance for this single application,
        # reusing the same validators the batch data-quality pipeline uses
        # (see data_quality/validators.py) rather than a separate ad hoc
        # single-row check.
        with runner.step("validation"):
            row_df = pd.DataFrame([application.model_dump()])
            schema_issues = validate_schema(row_df)
            critical = [i for i in schema_issues if i.severity == "critical"]
            if critical:
                raise DataValidationError(f"Schema validation failed: {critical[0].message}")

        # 2. Data quality check -- domain business-rule validation (e.g.
        # loan_amount > 0, age >= 18) against this single row.
        with runner.step("data_quality_check"):
            dq_issues = validate_business_rules(row_df, default_credit_business_rules())
            critical_dq = [i for i in dq_issues if i.severity == "critical"]
            if critical_dq:
                raise DataValidationError(f"Data quality check failed: {critical_dq[0].message}")

        # 3. Fraud check -- PLACEHOLDER ARCHITECTURE, as explicitly scoped:
        # no fraud-detection model exists in this platform yet. This step
        # is a real extension point (see plugins/fraud_detection/), not a
        # fake pass-through -- it's recorded as "skipped" with an honest
        # reason rather than silently omitted or faked as "completed".
        #
        # Deliberately NOT `non_fatal=True`: a flagged fraud result must abort
        # the workflow the same way `validation`/`data_quality_check` do.
        # `non_fatal` is reserved for steps whose failure shouldn't block a
        # decision already made (`report_generation`, `notification`) --
        # letting a fraud flag through to scoring/decisioning would defeat
        # the point of the check entirely.
        if self.config_service.get("workflow.fraud_check_enabled"):
            with runner.step("fraud_check"):
                from plugins.fraud_detection import run_fraud_check

                fraud_result = run_fraud_check(application.model_dump())
                if fraud_result.get("flagged"):
                    raise DataValidationError(f"Fraud check flagged this application: {fraud_result.get('reason')}")
        else:
            runner.skip(
                "fraud_check",
                reason="workflow.fraud_check_enabled=False (placeholder architecture, not yet configured)",
            )

        # 4. AI Prediction -- the actual model call, reusing ScoringService
        # so a workflow-produced prediction is identical to (and persisted
        # the same way as) one produced via POST /predict directly.
        with runner.step("ai_prediction"):
            prediction_result = self.scoring_service.score_single(application, bundle, requested_by=requested_by)

        # 5. Business Rules -- may override the model's own decision.
        rule_context = {**application.model_dump(), **prediction_result}
        if self.config_service.get("feature_flags.rule_engine_enabled"):
            with runner.step("business_rules"):
                rule_result = self.rule_service.evaluate(rule_context)
        else:
            runner.skip("business_rules", reason="feature_flags.rule_engine_enabled=False")

        # 6. Manual Review -- triggered by an explicit rule match OR a
        # borderline model confidence band (configurable), unless manual
        # review is disabled entirely.
        with runner.step("manual_review"):
            if rule_result and rule_result["matched_rule_type"] == "manual_review":
                needs_manual_review = True
            elif self.config_service.get("workflow.manual_review_enabled") and not (
                rule_result and rule_result["has_override"]
            ):
                band = self.config_service.get("workflow.manual_review_band") or {}
                proba = prediction_result["predicted_probability"]
                if band.get("lower", 0) <= proba <= band.get("upper", 0):
                    needs_manual_review = True

        # 7. Decision -- a matched approval/rejection rule overrides the
        # model; otherwise a manual-review flag takes precedence over the
        # model's own decision; otherwise the model's decision stands.
        with runner.step("decision"):
            if rule_result and rule_result["matched_rule_type"] in ("approval", "rejection"):
                final_decision = "approved" if rule_result["matched_rule_type"] == "approval" else "rejected"
            elif needs_manual_review:
                final_decision = "manual_review"
            else:
                final_decision = prediction_result["decision"]

        # 8. Report generation -- a lightweight structured summary here
        # (not the full SHAP/LIME HTML report, which is comparatively
        # expensive to compute and available on demand via
        # POST /reports/explainability or POST /copilot/explain-prediction
        # for any prediction_id produced by this workflow).
        with runner.step("report_generation", non_fatal=True):
            report_summary = {
                "prediction_id": prediction_result["prediction_id"],
                "final_decision": final_decision,
                "predicted_probability": prediction_result["predicted_probability"],
                "rule_override_applied": bool(rule_result and rule_result["has_override"]),
            }

        # 9. Notification -- best-effort, never fails the workflow.
        with runner.step("notification", non_fatal=True):
            self.notification_service.notify(
                event_type="workflow.decision",
                recipient_user_id=requested_by,
                message=f"Application {application.customer_id or prediction_result['prediction_id']} "
                f"decisioned: {final_decision}",
                details=report_summary,
            )

        # 10. Audit logging -- one summary entry for the whole run, in
        # addition to the per-step entries already written by the
        # services this workflow called (score_single already logs
        # "predict"; rule/config services log their own writes).
        with runner.step("audit_logging"):
            self.audit_repo.log_action(
                user_id=requested_by,
                action="workflow.execute",
                resource=f"prediction:{prediction_result['prediction_id']}",
                details={"final_decision": final_decision, "steps": len(runner.run.steps)},
            )

        execution = WorkflowExecution(
            customer_id=application.customer_id,
            prediction_id=prediction_result["prediction_id"],
            status=runner.run.status,
            final_decision=final_decision,
            steps=runner.run.as_dicts(),
            requested_by=requested_by,
        )
        execution.completed_at = datetime.now(timezone.utc)
        execution = self.workflow_repo.add(execution)

        return {
            "execution_id": execution.id,
            "status": execution.status,
            "final_decision": final_decision,
            "prediction": prediction_result,
            "rule_evaluation": rule_result,
            "needs_manual_review": needs_manual_review,
            "steps": execution.steps,
        }
