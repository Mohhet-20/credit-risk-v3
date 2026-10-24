"""
AI Copilot: explain predictions, summarize dashboards/reports, and generate
an executive briefing -- all built on top of existing, already-computed
platform data (SHAP/LIME/counterfactual explanations, evaluation metrics,
monitoring stats, drift, data-quality scores, and AI
Governance status) via `explainability/copilot.py`'s deterministic
template-based narrative generation. See that module's docstring for why
this is intentionally NOT backed by an external LLM API.

RBAC: `/copilot/explain-prediction` mirrors `/explain`'s access level
(`loan_officer`+, a business workflow surface). The summary/briefing
endpoints mirror `/reports/*`'s access level (`risk_analyst`+, `auditor`,
`ml_engineer` -- the same audience as the underlying compliance/governance
reports they summarize).
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import (
    get_data_quality_service,
    get_governance_service,
    get_model_bundle,
    get_monitoring_service,
    require_role,
    require_roles,
)
from api.routers.explain import _build_explainer
from api.schemas import CreditApplicationRequest
from core.exceptions import ModelNotFoundError, ResourceNotFoundError
from core.security import roles_at_least
from explainability.copilot import (
    explain_prediction_for_copilot,
    generate_executive_summary,
    summarize_data_quality,
    summarize_governance,
    summarize_model_performance,
    summarize_monitoring,
)
from explainability.counterfactuals import generate_counterfactual
from explainability.narrative import explain_prediction_plain_english
from models.registry import ModelBundle
from reporting.evaluation_report import compute_evaluation_metrics
from services.data_quality_service import DataQualityService
from services.governance_service import GovernanceService
from services.monitoring_service import MonitoringService

router = APIRouter(prefix="/copilot", tags=["copilot"])

_SUMMARY_ROLES = (*roles_at_least("risk_analyst"), "auditor", "ml_engineer")


class CopilotResponse(BaseModel):
    """Uniform shape for every copilot answer -- a headline for a card/toast,
    a fuller narrative paragraph, a bulleted insight list, and concrete
    recommended actions. Consistent across explain/summarize/briefing so a
    single frontend "Copilot" component can render any of them."""

    headline: str
    narrative: str
    insights: list[str]
    recommended_actions: list[str]


@router.post("/explain-prediction", response_model=CopilotResponse)
def explain_prediction(
    application: CreditApplicationRequest,
    current_user=Depends(require_role("loan_officer")),
    bundle: ModelBundle = Depends(get_model_bundle),
):
    """The copilot's flagship capability: explain a single prediction in
    plain English, backed by the same SHAP local explanation and
    counterfactual "path to approval" search `/explain` and `/reports/
    explainability` already compute -- reused here, not recomputed with
    different logic, so the copilot's explanation and the compliance
    report's explanation can never silently disagree."""
    raw_df = pd.DataFrame([application.model_dump()])
    transformed = bundle.feature_pipeline.transform(raw_df)
    proba = float(bundle.model.predict_proba(transformed)[:, 1][0])
    threshold = bundle.metadata["decision_threshold"]

    explainer = _build_explainer(bundle)
    local = explainer.local_explanation(transformed)
    plain_text = explain_prediction_plain_english(local, proba, threshold)

    def predict_fn(df: pd.DataFrame):
        return bundle.model.predict_proba(df)[:, 1]

    cf = generate_counterfactual(predict_fn, bundle.feature_pipeline, raw_df, threshold)
    cf_changes = cf.changes if cf.changes else None

    result = explain_prediction_for_copilot(
        plain_english_text=plain_text,
        predicted_probability=proba,
        decision_threshold=threshold,
        shap_contributions=local["contributions"],
        counterfactual_changes=cf_changes,
    )
    return CopilotResponse(**result)


@router.get("/summary/model-performance", response_model=CopilotResponse)
def summary_model_performance(
    current_user=Depends(require_roles(*_SUMMARY_ROLES)),
    bundle: ModelBundle = Depends(get_model_bundle),
):
    metrics = compute_evaluation_metrics(bundle)
    result = summarize_model_performance(
        metrics,
        model_name=bundle.metadata["model_name"],
        model_version=bundle.metadata["version"],
    )
    return CopilotResponse(**result)


@router.get("/summary/monitoring", response_model=CopilotResponse)
def summary_monitoring(
    current_user=Depends(require_roles(*_SUMMARY_ROLES)),
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    stats = monitoring.get_stats()
    distribution = monitoring.get_distribution()
    drift = monitoring.get_drift()
    latency = monitoring.get_latency_stats()
    result = summarize_monitoring(stats, distribution, drift, latency)
    return CopilotResponse(**result)


@router.get("/summary/data-quality", response_model=CopilotResponse)
def summary_data_quality(
    current_user=Depends(require_roles(*_SUMMARY_ROLES)),
    service: DataQualityService = Depends(get_data_quality_service),
):
    report = service.get_latest()
    if report is None:
        raise ResourceNotFoundError(
            "No data quality report has been generated yet. POST /data-quality/run first.",
        )
    result = summarize_data_quality(report)
    return CopilotResponse(**result)


@router.get("/summary/governance", response_model=CopilotResponse)
def summary_governance(
    current_user=Depends(require_roles(*_SUMMARY_ROLES)),
    governance_service: GovernanceService = Depends(get_governance_service),
):
    """Executive-readable summary of AI Governance status (champion model,
    active business rules, recent workflow outcomes, data-quality status)
    -- the "Executive Insights" narrative layer over `GET /reports/
    compliance-dashboard`'s raw numbers. See `summarize_governance()`."""
    dashboard = governance_service.compliance_dashboard()
    result = summarize_governance(dashboard)
    return CopilotResponse(**result)


@router.get("/executive-briefing", response_model=CopilotResponse)
def executive_briefing(
    current_user=Depends(require_roles(*_SUMMARY_ROLES)),
    bundle: ModelBundle = Depends(get_model_bundle),
    monitoring: MonitoringService = Depends(get_monitoring_service),
    dq_service: DataQualityService = Depends(get_data_quality_service),
    governance_service: GovernanceService = Depends(get_governance_service),
):
    """The single "ask the copilot for today's status" endpoint -- combines
    model performance, operations/monitoring, data quality, and
    business-rule/workflow governance status
    into one briefing. Degrades gracefully: a section with no data yet
    (e.g. no data-quality run performed) is simply omitted rather than
    failing the whole request, since a briefing with three out of four
    sections is far more useful than a 404."""
    model_summary = None
    try:
        metrics = compute_evaluation_metrics(bundle)
        model_summary = summarize_model_performance(
            metrics,
            model_name=bundle.metadata["model_name"],
            model_version=bundle.metadata["version"],
        )
    except (ModelNotFoundError, FileNotFoundError):
        pass

    monitoring_summary = summarize_monitoring(
        monitoring.get_stats(),
        monitoring.get_distribution(),
        monitoring.get_drift(),
        monitoring.get_latency_stats(),
    )

    dq_report = dq_service.get_latest()
    dq_summary = summarize_data_quality(dq_report) if dq_report is not None else None

    governance_summary = summarize_governance(governance_service.compliance_dashboard())

    result = generate_executive_summary(model_summary, monitoring_summary, dq_summary, governance_summary)
    return CopilotResponse(**result)
