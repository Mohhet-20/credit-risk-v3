"""
Model registry endpoints: list versions, view comparison table, promote,
rollback, and hot-reload the in-process model cache.

Design decision: `promote`/`rollback` no longer catch a bare `Exception` and
manually re-raise `HTTPException(404, ...)` -- `ModelGovernanceService` (and
the `ModelRegistry` it wraps) raises the typed `ModelNotFoundError`, and
`api/app.py` has a dedicated handler translating it to 404. Letting the
domain exception propagate instead of catching-and-rewrapping keeps this
router's handlers free of error-translation logic entirely.

Security fix (this pass): `GET /versions` and `GET /comparison` previously
had NO authentication dependency at all -- fully unauthenticated model
metadata exposure. Both now require at least `viewer` (any authenticated
user), the least-privileged fix that closes the gap without narrowing who
could already see this data legitimately.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_model_governance_service, require_role, require_roles
from api.schemas import ModelVersionOut, PromoteModelRequest
from configs.settings import get_settings
from core.security import roles_at_least
from database.models import User
from services.model_governance_service import ModelGovernanceService

router = APIRouter(prefix="/models", tags=["models"])

# `ml_engineer` owns model governance day to day (promoting a newly-trained
# challenger, rolling back a regression) without needing user-management
# power -- see `core/constants.py::UserRole`'s docstring on lateral roles.
_MODEL_WRITE_ROLES = (*roles_at_least("admin"), "ml_engineer")


@router.get("/versions", response_model=list[ModelVersionOut], dependencies=[Depends(require_role("viewer"))])
def list_versions(governance: ModelGovernanceService = Depends(get_model_governance_service)):
    return governance.list_versions()


@router.get("/comparison", dependencies=[Depends(require_role("viewer"))])
def model_comparison(governance: ModelGovernanceService = Depends(get_model_governance_service)):
    return governance.get_comparison()


@router.get("/validation-report", dependencies=[Depends(require_role("viewer"))])
def validation_report():
    """The champion's independent-validation evidence, written by the training
    pipeline: out-of-time discrimination (AUC/Gini/KS with bootstrap CIs),
    calibration, the gains table, the cost-sensitive threshold and the
    fair-lending (four-fifths rule) review."""
    path = get_settings().reports_dir / "model_validation.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No validation report yet -- run the training pipeline.")
    return json.loads(path.read_text())


@router.post("/promote", response_model=ModelVersionOut)
def promote_model(
    payload: PromoteModelRequest,
    current_user: User = Depends(require_roles(*_MODEL_WRITE_ROLES)),
    governance: ModelGovernanceService = Depends(get_model_governance_service),
):
    bundle = governance.promote(payload.version, actor_user_id=current_user.id)
    return ModelVersionOut(
        version=payload.version,
        model_name=bundle.metadata["model_name"],
        registered_at=bundle.metadata["registered_at"],
        decision_threshold=bundle.metadata.get("decision_threshold"),
        test_metrics=bundle.metadata.get("test_metrics"),
        is_champion=True,
    )


@router.post("/rollback", response_model=ModelVersionOut)
def rollback_model(
    payload: PromoteModelRequest,
    current_user: User = Depends(require_roles(*_MODEL_WRITE_ROLES)),
    governance: ModelGovernanceService = Depends(get_model_governance_service),
):
    bundle = governance.rollback(payload.version, actor_user_id=current_user.id)
    return ModelVersionOut(
        version=payload.version,
        model_name=bundle.metadata["model_name"],
        registered_at=bundle.metadata["registered_at"],
        decision_threshold=bundle.metadata.get("decision_threshold"),
        test_metrics=bundle.metadata.get("test_metrics"),
        is_champion=True,
    )


@router.post("/reload-cache", status_code=204)
def reload_model_cache(
    current_user: User = Depends(require_roles(*_MODEL_WRITE_ROLES)),
    governance: ModelGovernanceService = Depends(get_model_governance_service),
):
    governance.reload_cache()
