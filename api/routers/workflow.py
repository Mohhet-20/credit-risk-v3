"""
Enterprise Workflow Engine endpoints: the full governed credit-decisioning
pipeline (validation -> data quality check -> fraud check -> AI prediction
-> business rules -> manual review -> decision -> report generation ->
notification -> audit logging) -- see `workflows/engine.py` for the
step-timing scaffolding and `services/workflow_service.py` for the
orchestration this router calls.

This is ADDITIVE alongside `POST /predict` (see
`services/workflow_service.py`'s module docstring) -- existing
integrations calling `/predict` directly are completely unaffected by
this router's existence.

RBAC: executing the workflow is `loan_officer`-or-above (same tier that
can call `/predict` and `/explain` directly). Reading execution history/
stats additionally admits `auditor` and `ml_engineer`, the same
lateral-role pattern used elsewhere for compliance/governance reads.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_model_bundle, get_workflow_service, require_role, require_roles
from api.schemas import CreditApplicationRequest, WorkflowExecuteResponse, WorkflowExecutionOut
from core.security import roles_at_least
from database.models import User
from models.registry import ModelBundle
from services.workflow_service import WorkflowService
from workflows.engine import STANDARD_WORKFLOW_STEPS

router = APIRouter(prefix="/workflow", tags=["workflow"])

_READ_ROLES = (*roles_at_least("risk_analyst"), "auditor", "ml_engineer")


@router.get("/definition")
def workflow_definition(current_user: User = Depends(require_role("loan_officer"))):
    """The standard workflow's ordered step list -- lets a client render a
    pipeline progress UI without hard-coding the step names."""
    return {"steps": list(STANDARD_WORKFLOW_STEPS)}


@router.post("/execute", response_model=WorkflowExecuteResponse)
def execute_workflow(
    application: CreditApplicationRequest,
    current_user: User = Depends(require_role("loan_officer")),
    bundle: ModelBundle = Depends(get_model_bundle),
    service: WorkflowService = Depends(get_workflow_service),
):
    return service.execute(application, bundle, requested_by=current_user.id)


@router.get("/executions", response_model=list[WorkflowExecutionOut])
def list_workflow_executions(
    limit: int = 100,
    current_user: User = Depends(require_roles(*_READ_ROLES)),
    service: WorkflowService = Depends(get_workflow_service),
):
    return service.list_executions(limit=min(limit, 500))


@router.get("/executions/{execution_id}", response_model=WorkflowExecutionOut)
def get_workflow_execution(
    execution_id: str,
    current_user: User = Depends(require_roles(*_READ_ROLES)),
    service: WorkflowService = Depends(get_workflow_service),
):
    return service.get_execution(execution_id)


@router.get("/stats")
def workflow_stats(
    days: int = 30,
    current_user: User = Depends(require_roles(*_READ_ROLES)),
    service: WorkflowService = Depends(get_workflow_service),
):
    return service.stats(days=days)
