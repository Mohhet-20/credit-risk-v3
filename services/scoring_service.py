"""
Prediction / scoring business logic: single, batch (sync), batch
(async/background), and prediction history lookup.

Design decision: `_JOB_REGISTRY` stays a module-level dict rather than an
instance attribute on `ScoringService`, because a fresh `ScoringService` is
constructed per-request via FastAPI's DI (see `api/dependencies.py`) -- an
instance-level registry would lose every in-flight job between requests.
This is the same in-process-registry tradeoff documented in the original
`api/routers/scoring.py`: appropriate at this platform's scale, with the
`BatchJobStatus` polling contract already shaped so swapping in a real task
queue (Celery + Redis/RabbitMQ) later is a backend-only change.

Reliability (this pass): `_JOB_REGISTRY` previously grew without bound --
every `POST /predict/batch/async` call added an entry and nothing ever
removed one, a slow memory leak over the life of a long-running API
process. `queue_batch_job()` now sweeps entries older than `JOB_TTL_HOURS`
(and, as a hard backstop, oldest-first once the registry exceeds
`MAX_JOB_REGISTRY_SIZE`) on every call -- cheap, since it only runs when a
new job is queued, and it never evicts a `queued`/`running` job regardless
of age, only ones that have actually finished (`completed`/`failed`).
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd

from api.schemas import BatchJobStatus, CreditApplicationRequest
from core.constants import RISK_BAND_THRESHOLDS, RiskBand
from core.exceptions import CreditRiskPlatformError
from core.logging import get_logger
from core.scorecard import pd_to_score
from database.session import session_scope
from models.registry import ModelBundle
from repositories.audit_log_repository import AuditLogRepository
from repositories.prediction_repository import PredictionRepository

logger = get_logger(__name__)

MAX_SYNC_BATCH_SIZE = 500
JOB_TTL_HOURS = 24
MAX_JOB_REGISTRY_SIZE = 1000
_JOB_REGISTRY: dict[str, BatchJobStatus] = {}


def _evict_stale_jobs() -> None:
    """Removes finished (`completed`/`failed`) jobs older than
    `JOB_TTL_HOURS`, then -- as a hard backstop against a pathological burst
    of jobs within the TTL window -- evicts the oldest finished jobs until
    the registry is back under `MAX_JOB_REGISTRY_SIZE`. `queued`/`running`
    jobs are never evicted by either path; only age/size-bounding of jobs
    that have already reached a terminal state."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=JOB_TTL_HOURS)
    finished = {"completed", "failed"}
    for job_id, job in list(_JOB_REGISTRY.items()):
        if job.status in finished and job.created_at < cutoff:
            del _JOB_REGISTRY[job_id]

    if len(_JOB_REGISTRY) > MAX_JOB_REGISTRY_SIZE:
        evictable = sorted(
            (j for j in _JOB_REGISTRY.values() if j.status in finished),
            key=lambda j: j.created_at,
        )
        for job in evictable[: len(_JOB_REGISTRY) - MAX_JOB_REGISTRY_SIZE]:
            del _JOB_REGISTRY[job.job_id]


class BatchTooLargeError(CreditRiskPlatformError):
    """Raised when a synchronous batch request exceeds MAX_SYNC_BATCH_SIZE."""


def _risk_band(probability: float) -> str:
    for band, (lo, hi) in RISK_BAND_THRESHOLDS.items():
        if lo <= probability < hi:
            return band.value
    return RiskBand.VERY_HIGH.value


def score_application(application: CreditApplicationRequest, bundle: ModelBundle) -> tuple[float, str, str, float]:
    raw_df = pd.DataFrame([application.model_dump()])
    transformed = bundle.feature_pipeline.transform(raw_df)
    proba = float(bundle.model.predict_proba(transformed)[:, 1][0])
    threshold = bundle.metadata["decision_threshold"]
    decision = "rejected" if proba >= threshold else "approved"
    return proba, _risk_band(proba), decision, threshold


def decline_reasons(application: CreditApplicationRequest, bundle: ModelBundle) -> list[dict]:
    """Adverse-action reasons for a declined application. A failure here must
    not fail the credit decision itself, but it is logged loudly: a decline
    without reasons is a compliance gap someone has to close by hand."""
    from explainability import explainer_cache
    from explainability.adverse_action import adverse_action_reasons

    try:
        transformed = bundle.feature_pipeline.transform(pd.DataFrame([application.model_dump()]))
        local = explainer_cache.get_explainer(bundle).local_explanation(transformed)
        return adverse_action_reasons(local["contributions"])
    except Exception:  # noqa: BLE001
        logger.exception(
            "Adverse-action reason generation failed",
            extra={
                "context": {"customer_id": application.customer_id, "model_version": bundle.metadata.get("version")}
            },
        )
        return []


class ScoringService:
    def __init__(self, prediction_repo: PredictionRepository, audit_repo: AuditLogRepository):
        self.prediction_repo = prediction_repo
        self.audit_repo = audit_repo

    def score_single(self, application: CreditApplicationRequest, bundle: ModelBundle, *, requested_by: str) -> dict:
        start = time.perf_counter()
        proba, band, decision, threshold = score_application(application, bundle)
        reasons = decline_reasons(application, bundle) if decision == "rejected" else []
        latency_ms = (time.perf_counter() - start) * 1000

        record = self.prediction_repo.create(
            customer_id=application.customer_id,
            input_features=application.model_dump(),
            predicted_probability=proba,
            risk_band=band,
            decision=decision,
            decision_threshold=threshold,
            model_version=bundle.metadata["version"],
            model_name=bundle.metadata["model_name"],
            latency_ms=latency_ms,
            request_source="api",
            requested_by=requested_by,
        )
        self.audit_repo.log_action(
            user_id=requested_by,
            action="predict",
            resource=f"prediction:{record.id}",
            details={"decision": decision, "reason_codes": [r["code"] for r in reasons]} if reasons else None,
        )

        return {
            "prediction_id": record.id,
            "customer_id": application.customer_id,
            "predicted_probability": proba,
            "credit_score": pd_to_score(proba),
            "risk_band": band,
            "decision": decision,
            "decision_threshold": threshold,
            "adverse_action_reasons": reasons,
            "model_name": bundle.metadata["model_name"],
            "model_version": bundle.metadata["version"],
            "latency_ms": latency_ms,
        }

    def score_batch(
        self,
        applications: list[CreditApplicationRequest],
        bundle: ModelBundle,
        *,
        requested_by: str,
    ) -> list[dict]:
        if len(applications) > MAX_SYNC_BATCH_SIZE:
            raise BatchTooLargeError(
                f"Batch too large for synchronous scoring (max {MAX_SYNC_BATCH_SIZE}); use /predict/batch/async.",
            )
        results = []
        for application in applications:
            results.append(self.score_single(application, bundle, requested_by=requested_by))
        self.audit_repo.log_action(user_id=requested_by, action="predict_batch", details={"n": len(results)})
        return results

    def get_history(self, *, customer_id: str | None = None, limit: int = 200) -> list[dict]:
        records = self.prediction_repo.list_predictions(limit=limit, customer_id=customer_id)
        return [
            {
                "prediction_id": r.id,
                "customer_id": r.customer_id,
                "predicted_probability": r.predicted_probability,
                "risk_band": r.risk_band,
                "decision": r.decision,
                "model_name": r.model_name,
                "model_version": r.model_version,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]

    @staticmethod
    def queue_batch_job(applications: list[CreditApplicationRequest]) -> BatchJobStatus:
        _evict_stale_jobs()
        job_id = str(uuid.uuid4())
        job = BatchJobStatus(job_id=job_id, status="queued", n_total=len(applications), n_processed=0)
        _JOB_REGISTRY[job_id] = job
        return job

    @staticmethod
    def get_job_status(job_id: str) -> BatchJobStatus | None:
        return _JOB_REGISTRY.get(job_id)

    @staticmethod
    def run_async_batch(job_id: str, applications: list[CreditApplicationRequest], user_id: str) -> None:
        """Executed by FastAPI's `BackgroundTasks` -- runs outside the request's
        DB session, so it opens its own via `session_scope()`."""
        from configs.settings import get_settings
        from models.registry import ModelRegistry

        job = _JOB_REGISTRY[job_id]
        job.status = "running"
        try:
            settings = get_settings()
            registry = ModelRegistry(settings.model_artifact_dir)
            bundle = registry.load_champion()

            rows = []
            with session_scope() as db:
                prediction_repo = PredictionRepository(db)
                for application in applications:
                    proba, band, decision, threshold = score_application(application, bundle)
                    record = prediction_repo.create(
                        customer_id=application.customer_id,
                        input_features=application.model_dump(),
                        predicted_probability=proba,
                        risk_band=band,
                        decision=decision,
                        decision_threshold=threshold,
                        model_version=bundle.metadata["version"],
                        model_name=bundle.metadata["model_name"],
                        latency_ms=None,
                        request_source="batch_async",
                        requested_by=user_id,
                    )
                    rows.append(
                        {
                            "prediction_id": record.id,
                            "customer_id": application.customer_id,
                            "predicted_probability": proba,
                            "risk_band": band,
                            "decision": decision,
                        }
                    )
                    job.n_processed += 1

            result_dir = settings.reports_dir / "batch_jobs"
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / f"{job_id}.csv"
            pd.DataFrame(rows).to_csv(result_path, index=False)

            job.status = "completed"
            job.result_path = str(result_path)
        except Exception:
            job.status = "failed"
            # Runs inside a FastAPI `BackgroundTasks` callback, outside request
            # scope -- without this, the failure is only visible via Starlette's
            # default background-task exception handling, never through this
            # app's structured JSON logging/correlation-ID pipeline like every
            # other failure path.
            logger.error(
                f"Async batch job {job_id} failed",
                extra={"context": {"job_id": job_id, "n_total": job.n_total, "n_processed": job.n_processed}},
                exc_info=True,
            )
            raise
