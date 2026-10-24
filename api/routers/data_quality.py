"""
Data quality endpoints: run a fresh validation + profiling + scoring pass
against the reference dataset, fetch the latest persisted result, or page
through history.

Design decision: like `/monitoring/drift` and `/explain`'s background
sample, this router defaults to `data/raw/credit_applications.csv` (the
reference dataset) rather than accepting an arbitrary file upload -- chunked
multi-format file ingestion (CSV/Excel/JSON/ZIP) was explicitly out of scope
for this pass (see the data-platform architecture note in
`docs/ARCHITECTURE.md`). `POST /data-quality/run` is the one mutating
endpoint here (it persists a new report row); the other two are pure reads.

RBAC (this pass): `ml_engineer` can trigger a run (they own the data feeding
the training pipeline); `auditor` gets read-only access to the latest report
and history (exactly the kind of compliance artifact an auditor role exists
to review) but never `/run` itself, since triggering a mutating pipeline
action isn't a read-only role's job. See `core/constants.py::UserRole`'s
docstring on lateral roles.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends

from api.dependencies import get_data_quality_service, require_roles
from configs.settings import get_settings
from core.constants import TARGET_COLUMN
from core.exceptions import ResourceNotFoundError
from core.pagination import PaginationParams, pagination_params
from core.security import roles_at_least
from services.data_quality_service import DataQualityService

router = APIRouter(prefix="/data-quality", tags=["data-quality"])

_RUN_ROLES = (*roles_at_least("risk_analyst"), "ml_engineer")
_READ_ROLES = (*roles_at_least("risk_analyst"), "ml_engineer", "auditor")


def _load_reference_dataset() -> tuple[pd.DataFrame, pd.Series | None, str]:
    settings = get_settings()
    path = settings.data_raw_dir / "credit_applications.csv"
    df = pd.read_csv(path)
    target = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
    features = df.drop(columns=[TARGET_COLUMN]) if target is not None else df
    return features, target, str(path)


@router.post("/run")
def run_data_quality_check(
    current_user=Depends(require_roles(*_RUN_ROLES)),
    service: DataQualityService = Depends(get_data_quality_service),
):
    features, target, path = _load_reference_dataset()
    return service.run_and_persist(
        features,
        target=target,
        outlier_columns=list(features.select_dtypes(include="number").columns),
        dataset_path=path,
        dataset_label="reference",
    )


@router.get("/latest")
def latest_data_quality_report(
    current_user=Depends(require_roles(*_READ_ROLES)),
    service: DataQualityService = Depends(get_data_quality_service),
):
    result = service.get_latest()
    if result is None:
        raise ResourceNotFoundError("No data quality report has been generated yet. POST /data-quality/run first.")
    return result


@router.get("/history")
def data_quality_history(
    params: PaginationParams = Depends(pagination_params),
    current_user=Depends(require_roles(*_READ_ROLES)),
    service: DataQualityService = Depends(get_data_quality_service),
):
    return service.list_history(params)
