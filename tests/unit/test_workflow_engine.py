"""Unit tests for the pure Workflow Engine step-timing scaffolding
(`workflows/engine.py`) -- completed/failed/skipped step recording,
non-fatal exception suppression, and overall run status derivation. No
DB, no FastAPI, no service layer."""

from __future__ import annotations

import pytest

from workflows.engine import STANDARD_WORKFLOW_STEPS, WorkflowRunner


def test_standard_workflow_steps_is_the_documented_order():
    assert STANDARD_WORKFLOW_STEPS == (
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


def test_completed_step_is_recorded_with_timing():
    runner = WorkflowRunner()
    with runner.step("validation"):
        pass
    assert len(runner.run.steps) == 1
    step = runner.run.steps[0]
    assert step.name == "validation"
    assert step.status == "completed"
    assert step.duration_ms >= 0
    assert step.detail is None


def test_fatal_step_failure_is_recorded_and_reraised():
    runner = WorkflowRunner()
    with pytest.raises(ValueError):
        with runner.step("validation"):
            raise ValueError("schema invalid")
    assert len(runner.run.steps) == 1
    assert runner.run.steps[0].status == "failed"
    assert runner.run.steps[0].detail == "schema invalid"


def test_non_fatal_step_failure_is_recorded_but_swallowed():
    runner = WorkflowRunner()
    with runner.step("notification", non_fatal=True):
        raise RuntimeError("notification provider unreachable")
    assert len(runner.run.steps) == 1
    assert runner.run.steps[0].status == "failed"
    assert runner.run.steps[0].detail == "notification provider unreachable"


def test_skip_records_a_skipped_step_with_reason():
    runner = WorkflowRunner()
    runner.skip("fraud_check", reason="workflow.fraud_check_enabled=False")
    assert len(runner.run.steps) == 1
    assert runner.run.steps[0].status == "skipped"
    assert runner.run.steps[0].detail == "workflow.fraud_check_enabled=False"


def test_run_status_is_completed_when_no_step_failed():
    runner = WorkflowRunner()
    with runner.step("validation"):
        pass
    runner.skip("fraud_check", reason="disabled")
    assert runner.run.status == "completed"


def test_run_status_is_failed_if_any_step_failed_even_if_non_fatal():
    """A non-fatal step failure still means the RUN honestly reports
    'failed' -- 'non_fatal' only controls whether the exception aborts the
    workflow, not whether the record is honest about what happened."""
    runner = WorkflowRunner()
    with runner.step("ai_prediction"):
        pass
    with runner.step("notification", non_fatal=True):
        raise RuntimeError("boom")
    assert runner.run.status == "failed"


def test_as_dicts_serializes_every_step():
    runner = WorkflowRunner()
    with runner.step("validation"):
        pass
    runner.skip("fraud_check", reason="disabled")
    dicts = runner.run.as_dicts()
    assert len(dicts) == 2
    assert {"name", "status", "started_at", "completed_at", "duration_ms", "detail"} <= dicts[0].keys()
