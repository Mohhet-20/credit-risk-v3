"""
Monitoring endpoints: prediction distribution, confidence, latency, error
rates, and data/feature drift against the training reference window.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_monitoring_service, require_role, require_roles
from api.schemas import MonitoringStats
from core.security import roles_at_least
from services.monitoring_service import MonitoringService

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/stats", response_model=MonitoringStats)
def prediction_stats(
    days: int = 30,
    current_user=Depends(require_role("viewer")),
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    return MonitoringStats(**monitoring.get_stats(days=days))


@router.get("/prediction-distribution")
def prediction_distribution(
    days: int = 30,
    current_user=Depends(require_role("viewer")),
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    return monitoring.get_distribution(days=days)


@router.get("/drift")
def feature_drift(
    # `auditor` gets read access here too -- drift status is exactly what a
    # compliance review would want to check. See
    # core/constants.py::UserRole's docstring on lateral roles.
    current_user=Depends(require_roles(*roles_at_least("risk_analyst"), "auditor")),
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    return monitoring.get_drift()


@router.get("/latency")
def latency_stats(
    current_user=Depends(require_role("viewer")),
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    return monitoring.get_latency_stats()
