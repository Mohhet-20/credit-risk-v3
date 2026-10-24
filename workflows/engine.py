"""
Enterprise Workflow Engine step definitions -- the standard credit-risk
pipeline expressed as an explicit, named, ordered sequence rather than an
implicit call chain buried inside a router handler:

    Validation -> Data Quality Check -> Fraud Check -> AI Prediction ->
    Business Rules -> Manual Review -> Decision -> Report Generation ->
    Notification -> Audit Logging

Design decision: this module defines the step NAMES and a small `StepResult`
/`WorkflowRunner` helper for timing and recording each step -- the actual
work each step does lives in `services/workflow_service.py`, which wires in
the real services (`ScoringService`, `RuleService`, `DataQualityService`,
etc.). Splitting "what steps exist, in what order, and how they're timed/
recorded" (this module, easily unit-testable, zero DB/service
dependencies) from "what each step actually calls" (the service, which
needs real dependencies) mirrors the same separation `rules/engine.py`
draws between pure evaluation logic and the DB-backed service around it.

Design decision: the step list is a plain ordered tuple, not a
DB-configurable table (unlike `BusinessRule`, which genuinely needs to be
admin-editable without a deploy). Reordering or disabling *individual*
pipeline stages (e.g. "skip fraud check for now") is deliberately a
`SystemConfig` flag the workflow service reads (see `services.
config_service.DEFAULT_CONFIG`'s `workflow.*` keys), not a step reordering
-- the STAGES of a regulated credit decision pipeline are a compliance
concern (you don't want an admin able to silently reorder "business rules"
to run after "decision" from a config screen), while whether an optional
stage runs at all is a legitimate operational toggle.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

STANDARD_WORKFLOW_STEPS = (
    "validation",
    "data_quality_check",
    "fraud_check",
    "ai_prediction",
    "business_rules",
    "manual_review",
    "decision",
    "report_generation",
    "notification",
    "audit_logging",
)


@dataclass
class StepResult:
    name: str
    status: str  # completed | failed | skipped
    started_at: str
    completed_at: str
    duration_ms: float
    detail: str | None = None


@dataclass
class WorkflowRun:
    steps: list[StepResult] = field(default_factory=list)
    context: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if any(s.status == "failed" for s in self.steps):
            return "failed"
        return "completed"

    def as_dicts(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "status": s.status,
                "started_at": s.started_at,
                "completed_at": s.completed_at,
                "duration_ms": s.duration_ms,
                "detail": s.detail,
            }
            for s in self.steps
        ]


class WorkflowRunner:
    """Records each step's timing/outcome into a `WorkflowRun` as it
    executes -- used by `WorkflowService.execute()` as:

        run = WorkflowRunner()
        with run.step("validation"):
            do_validation()
        with run.step("ai_prediction"):
            result = do_prediction()

    A step that raises is recorded as `status="failed"` with the
    exception's message as `detail`, and the exception is RE-RAISED (the
    runner records failures, it doesn't swallow them) -- `WorkflowService`
    decides whether a given step's failure should abort the whole run or
    be treated as non-fatal (e.g. a report-generation failure shouldn't
    block a decision that's already been made)."""

    def __init__(self) -> None:
        self.run = WorkflowRun()

    @contextmanager
    def step(self, name: str, *, non_fatal: bool = False):
        started = datetime.now(timezone.utc)
        start_perf = time.perf_counter()
        try:
            yield
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any step failure must be recorded
            self._record(name, "failed", started, start_perf, detail=str(exc))
            if non_fatal:
                return
            raise
        else:
            self._record(name, "completed", started, start_perf)

    def skip(self, name: str, *, reason: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.run.steps.append(
            StepResult(name=name, status="skipped", started_at=now, completed_at=now, duration_ms=0.0, detail=reason)
        )

    def _record(self, name: str, status: str, started: datetime, start_perf: float, detail: str | None = None) -> None:
        completed = datetime.now(timezone.utc)
        duration_ms = (time.perf_counter() - start_perf) * 1000
        self.run.steps.append(
            StepResult(
                name=name,
                status=status,
                started_at=started.isoformat(),
                completed_at=completed.isoformat(),
                duration_ms=round(duration_ms, 2),
                detail=detail,
            )
        )
