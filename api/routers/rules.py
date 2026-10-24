"""
Business Rule Engine endpoints: admin/risk-analyst-editable approval,
rejection, and manual-review rules that can override the AI model's
decision -- see `rules/engine.py` for the evaluation grammar and
`services/rule_service.py` for the CRUD + versioning + audit-log layer
this router calls.

Design decision: rule conditions are a small closed JSON grammar (field/
operator/value, with `all`/`any` composites), never a free-text expression
evaluated at runtime -- see `rules/engine.py`'s module docstring. This
closes off arbitrary code execution as an attack vector via an admin-
editable rule, at the cost of expressiveness a full expression language
would have; that tradeoff is intentional for a regulated credit-risk
product.

RBAC: rule management (create/update/activate/deactivate/simulate) is
`risk_analyst`-or-above -- these are the domain experts who own approval
policy. Reads additionally admit `auditor` and `ml_engineer` (the same
lateral-role pattern used throughout `api/routers/reports.py` and
`api/routers/data_quality.py`), since rule configuration is exactly the
kind of compliance artifact those roles exist to review.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_rule_service, require_roles
from api.schemas import BusinessRuleCreate, BusinessRuleOut, BusinessRuleUpdate, RuleSimulateRequest
from core.security import roles_at_least
from database.models import User
from services.rule_service import RuleService

router = APIRouter(prefix="/rules", tags=["rules"])

_MANAGE_ROLES = roles_at_least("risk_analyst")
_READ_ROLES = (*roles_at_least("risk_analyst"), "auditor", "ml_engineer")


@router.post("", response_model=BusinessRuleOut, status_code=201)
def create_rule(
    payload: BusinessRuleCreate,
    current_user: User = Depends(require_roles(*_MANAGE_ROLES)),
    service: RuleService = Depends(get_rule_service),
):
    return service.create_rule(
        name=payload.name,
        description=payload.description,
        rule_type=payload.rule_type,
        condition=payload.condition,
        priority=payload.priority,
        actor_user_id=current_user.id,
    )


@router.get("", response_model=list[BusinessRuleOut])
def list_rules(
    rule_type: str | None = None,
    current_user: User = Depends(require_roles(*_READ_ROLES)),
    service: RuleService = Depends(get_rule_service),
):
    return service.list_rules(rule_type=rule_type)


@router.get("/{rule_id}", response_model=BusinessRuleOut)
def get_rule(
    rule_id: str,
    current_user: User = Depends(require_roles(*_READ_ROLES)),
    service: RuleService = Depends(get_rule_service),
):
    return service.get_rule(rule_id)


@router.put("/{rule_id}", response_model=BusinessRuleOut)
def update_rule(
    rule_id: str,
    payload: BusinessRuleUpdate,
    current_user: User = Depends(require_roles(*_MANAGE_ROLES)),
    service: RuleService = Depends(get_rule_service),
):
    return service.update_rule(
        rule_id,
        actor_user_id=current_user.id,
        name=payload.name,
        description=payload.description,
        rule_type=payload.rule_type,
        condition=payload.condition,
        priority=payload.priority,
    )


@router.post("/{rule_id}/activate", response_model=BusinessRuleOut)
def activate_rule(
    rule_id: str,
    current_user: User = Depends(require_roles(*_MANAGE_ROLES)),
    service: RuleService = Depends(get_rule_service),
):
    return service.activate_rule(rule_id, actor_user_id=current_user.id)


@router.post("/{rule_id}/deactivate", response_model=BusinessRuleOut)
def deactivate_rule(
    rule_id: str,
    current_user: User = Depends(require_roles(*_MANAGE_ROLES)),
    service: RuleService = Depends(get_rule_service),
):
    return service.deactivate_rule(rule_id, actor_user_id=current_user.id)


@router.post("/simulate")
def simulate_rules(
    payload: RuleSimulateRequest,
    current_user: User = Depends(require_roles(*_MANAGE_ROLES)),
    service: RuleService = Depends(get_rule_service),
):
    """Dry-runs rules against a batch of hypothetical applicant contexts
    without persisting anything -- lets a risk analyst check "what would
    change" before activating a new or edited rule (or before activating
    a candidate rule at all, via `rule_ids` naming not-yet-active rules)."""
    return service.simulate(payload.contexts, rule_ids=payload.rule_ids)
