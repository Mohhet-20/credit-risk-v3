"""
Unit tests for the three governance repositories (`BusinessRuleRepository`,
`WorkflowExecutionRepository`, `SystemConfigRepository`), exercised
directly against an in-memory SQLite session -- same pattern as
`tests/unit/test_repositories.py`, kept in a separate file so the
rules/workflow/config subsystem's tests are easy to locate as a unit.
"""

from __future__ import annotations

from database.models import BusinessRule, WorkflowExecution
from repositories.business_rule_repository import BusinessRuleRepository
from repositories.system_config_repository import SystemConfigRepository
from repositories.workflow_execution_repository import WorkflowExecutionRepository


def test_business_rule_repository_list_active_orders_by_priority(db_session_factory):
    db = db_session_factory()
    repo = BusinessRuleRepository(db)
    repo.add(
        BusinessRule(
            name="low priority",
            rule_type="approval",
            condition={"field": "x", "operator": "eq", "value": 1},
            priority=200,
        )
    )
    repo.add(
        BusinessRule(
            name="high priority",
            rule_type="approval",
            condition={"field": "x", "operator": "eq", "value": 1},
            priority=10,
        )
    )
    repo.add(
        BusinessRule(
            name="inactive",
            rule_type="approval",
            condition={"field": "x", "operator": "eq", "value": 1},
            priority=5,
            is_active=False,
        )
    )

    active = repo.list_active()
    assert [r.name for r in active] == ["high priority", "low priority"]


def test_business_rule_repository_list_all_rules_filters_by_type(db_session_factory):
    db = db_session_factory()
    repo = BusinessRuleRepository(db)
    repo.add(BusinessRule(name="a", rule_type="approval", condition={"field": "x", "operator": "eq", "value": 1}))
    repo.add(BusinessRule(name="r", rule_type="rejection", condition={"field": "x", "operator": "eq", "value": 1}))

    approvals = repo.list_all_rules(rule_type="approval")
    assert len(approvals) == 1
    assert approvals[0].name == "a"


def test_business_rule_repository_update_bumps_version(db_session_factory):
    db = db_session_factory()
    repo = BusinessRuleRepository(db)
    rule = repo.add(
        BusinessRule(name="a", rule_type="approval", condition={"field": "x", "operator": "eq", "value": 1})
    )
    assert rule.version == 1

    updated = repo.update(rule, priority=5)
    assert updated.version == 2
    assert updated.priority == 5


def test_workflow_execution_repository_list_recent_orders_newest_first(db_session_factory):
    db = db_session_factory()
    repo = WorkflowExecutionRepository(db)
    first = repo.add(WorkflowExecution(status="completed", final_decision="approved"))
    second = repo.add(WorkflowExecution(status="completed", final_decision="rejected"))

    recent = repo.list_recent(limit=10)
    assert recent[0].id == second.id
    assert recent[1].id == first.id


def test_workflow_execution_repository_stats_last_n_days_counts_by_status(db_session_factory):
    db = db_session_factory()
    repo = WorkflowExecutionRepository(db)
    repo.add(WorkflowExecution(status="completed", final_decision="approved"))
    repo.add(WorkflowExecution(status="completed", final_decision="rejected"))
    repo.add(WorkflowExecution(status="failed"))

    stats = repo.stats_last_n_days(days=30)
    assert stats["total"] == 3
    assert stats["completed"] == 2
    assert stats["failed"] == 1


def test_system_config_repository_upsert_creates_then_updates(db_session_factory):
    db = db_session_factory()
    repo = SystemConfigRepository(db)

    created = repo.upsert(
        key="workflow.fraud_check_enabled", value=True, category="workflow", description="test", updated_by=None
    )
    assert created.value is True

    updated = repo.upsert(
        key="workflow.fraud_check_enabled", value=False, category="workflow", description="test", updated_by=None
    )
    assert updated.value is False
    assert updated.id == created.id  # upsert on existing key, not a new row


def test_system_config_repository_list_by_category(db_session_factory):
    db = db_session_factory()
    repo = SystemConfigRepository(db)
    repo.upsert(key="workflow.a", value=1, category="workflow", description=None, updated_by=None)
    repo.upsert(key="notifications.a", value=2, category="notifications", description=None, updated_by=None)

    workflow_only = repo.list_by_category("workflow")
    assert len(workflow_only) == 1
    assert workflow_only[0].key == "workflow.a"
