"""
Prediction endpoints: single, batch (sync), batch (async/background), and
prediction history lookup.

Design decision: this router is now a thin controller over
`services/scoring_service.py` -- see that module's docstring for the
async-batch job-registry design decision (still an in-process dict, now a
module-level detail of the service rather than the router).
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from api.dependencies import get_model_bundle, get_scoring_service, require_role
from api.schemas import (
    BatchJobStatus,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CreditApplicationRequest,
    PredictionResponse,
)
from core.rate_limit import limiter
from database.models import User
from models.registry import ModelBundle
from services.scoring_service import ScoringService

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictionResponse)
def predict_single(
    application: CreditApplicationRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
    current_user: User = Depends(require_role("loan_officer")),
    scoring_service: ScoringService = Depends(get_scoring_service),
):
    return scoring_service.score_single(application, bundle, requested_by=current_user.id)


@router.post("/batch", response_model=BatchPredictionResponse)
@limiter.limit("20/minute")  # stricter than the global default -- a synchronous batch call is expensive per request
def predict_batch(
    request: Request,
    payload: BatchPredictionRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
    current_user: User = Depends(require_role("loan_officer")),
    scoring_service: ScoringService = Depends(get_scoring_service),
):
    results = scoring_service.score_batch(payload.applications, bundle, requested_by=current_user.id)
    return BatchPredictionResponse(results=results, n_processed=len(results))


@router.post("/batch/async", response_model=BatchJobStatus, status_code=status.HTTP_202_ACCEPTED)
def predict_batch_async(
    payload: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role("loan_officer")),
):
    job = ScoringService.queue_batch_job(payload.applications)
    background_tasks.add_task(ScoringService.run_async_batch, job.job_id, payload.applications, current_user.id)
    return job


@router.get("/batch/status/{job_id}", response_model=BatchJobStatus)
def get_batch_status(job_id: str, current_user: User = Depends(require_role("loan_officer"))):
    job = ScoringService.get_job_status(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found.")
    return job


@router.get("/history")
def prediction_history(
    customer_id: str | None = None,
    limit: int = 200,
    current_user: User = Depends(require_role("viewer")),
    scoring_service: ScoringService = Depends(get_scoring_service),
):
    """Backs the frontend's Customer Search page — lists recent predictions,
    optionally filtered to a single customer_id."""
    return scoring_service.get_history(customer_id=customer_id, limit=limit)
