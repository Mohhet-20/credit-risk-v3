"""
Downloadable HTML reports: Model Comparison, Evaluation, Drift, Performance
Summary, Data Quality, and per-applicant Explainability (SHAP + LIME +
counterfactual).

Design decision: this router only ASSEMBLES data already produced by
existing services (`ModelGovernanceService`, `MonitoringService`,
`DataQualityService`) and existing explainability building blocks (reused
from `api/routers/explain.py` rather than duplicated) into `reporting/*.py`'s
HTML builders -- no new business logic lives here, consistent with the
"router = parse, call, shape response" convention used throughout
`api/routers/`.

RBAC (this pass): every GET report here (model comparison, evaluation,
drift, performance, data-quality) now also admits `auditor` and
`ml_engineer` alongside `risk_analyst`-or-above -- these are exactly the
compliance/model-governance artifacts those two lateral roles exist to
review. `POST /explainability` (per-applicant, business-workflow-facing,
not a compliance surface) is intentionally left at `loan_officer`-or-above
only. See `core/constants.py::UserRole`'s docstring on lateral roles.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, Response

from api.dependencies import (
    get_audit_service,
    get_data_quality_service,
    get_governance_service,
    get_model_bundle,
    get_model_governance_service,
    get_monitoring_service,
    get_scoring_service,
    require_role,
    require_roles,
)
from api.routers.explain import _build_explainer, _build_lime_explainer
from api.schemas import CreditApplicationRequest
from core.exceptions import ModelNotFoundError, ResourceNotFoundError
from core.pagination import pagination_params
from core.security import roles_at_least
from explainability.counterfactuals import generate_counterfactual
from explainability.narrative import explain_prediction_plain_english
from models.registry import ModelBundle
from reporting.data_card_report import build_data_card_html
from reporting.data_quality_report import build_data_quality_report_html
from reporting.drift_report import build_drift_report_html
from reporting.evaluation_report import build_evaluation_report_html, compute_evaluation_metrics
from reporting.explainability_report import build_explainability_report_html
from reporting.export import rows_to_csv_bytes, rows_to_excel_bytes
from reporting.model_card_report import build_model_card_html
from reporting.model_comparison_report import build_model_comparison_report_html
from reporting.performance_summary_report import build_performance_summary_html
from services.audit_service import AuditService
from services.data_quality_service import DataQualityService
from services.governance_service import GovernanceService
from services.model_governance_service import ModelGovernanceService
from services.monitoring_service import MonitoringService
from services.scoring_service import ScoringService

router = APIRouter(prefix="/reports", tags=["reports"])

_REPORT_READ_ROLES = (*roles_at_least("risk_analyst"), "auditor", "ml_engineer")


@router.get("/model-comparison", response_class=HTMLResponse)
def model_comparison_report(
    current_user=Depends(require_roles(*_REPORT_READ_ROLES)),
    governance: ModelGovernanceService = Depends(get_model_governance_service),
):
    try:
        comparison_rows = governance.get_comparison()
    except ModelNotFoundError:
        comparison_rows = []
    versions = governance.list_versions()
    champion = next((v["version"] for v in versions if v.get("is_champion")), None)
    return build_model_comparison_report_html(comparison_rows, versions, champion)


@router.get("/evaluation", response_class=HTMLResponse)
def evaluation_report(
    current_user=Depends(require_roles(*_REPORT_READ_ROLES)),
    bundle: ModelBundle = Depends(get_model_bundle),
):
    metrics = compute_evaluation_metrics(bundle)
    return build_evaluation_report_html(
        metrics,
        model_name=bundle.metadata["model_name"],
        model_version=bundle.metadata["version"],
    )


@router.get("/drift", response_class=HTMLResponse)
def drift_report(
    current_user=Depends(require_roles(*_REPORT_READ_ROLES)),
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    drift_result = monitoring.get_drift()
    return build_drift_report_html(drift_result)


@router.get("/performance-summary", response_class=HTMLResponse)
def performance_summary_report(
    current_user=Depends(require_roles(*_REPORT_READ_ROLES)),
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    stats = monitoring.get_stats()
    latency_stats = monitoring.get_latency_stats()
    return build_performance_summary_html(stats, latency_stats)


@router.get("/data-quality", response_class=HTMLResponse)
def data_quality_report(
    current_user=Depends(require_roles(*_REPORT_READ_ROLES)),
    service: DataQualityService = Depends(get_data_quality_service),
):
    """Renders the LATEST persisted data-quality report. Run
    `POST /data-quality/run` first if none exists yet -- this endpoint
    deliberately doesn't run a fresh check itself (that's a mutating,
    persistence-writing operation and belongs behind an explicit POST, not
    a GET a browser tab or crawler could trigger accidentally)."""
    result = service.get_latest()
    if result is None:
        raise ResourceNotFoundError(
            "No data quality report has been generated yet. POST /data-quality/run first.",
        )
    return build_data_quality_report_html(result)


@router.post("/explainability", response_class=HTMLResponse)
def explainability_report(
    application: CreditApplicationRequest,
    current_user=Depends(require_role("loan_officer")),
    bundle: ModelBundle = Depends(get_model_bundle),
):
    raw_df = pd.DataFrame([application.model_dump()])
    transformed = bundle.feature_pipeline.transform(raw_df)
    proba = float(bundle.model.predict_proba(transformed)[:, 1][0])

    shap_explainer = _build_explainer(bundle)
    shap_local = shap_explainer.local_explanation(transformed)
    plain_text = explain_prediction_plain_english(shap_local, proba, bundle.metadata["decision_threshold"])

    lime_explainer = _build_lime_explainer(bundle)

    def predict_fn(df: pd.DataFrame):
        return bundle.model.predict_proba(df)[:, 1]

    lime_result = lime_explainer.explain_row(predict_fn, transformed)

    cf = generate_counterfactual(predict_fn, bundle.feature_pipeline, raw_df, bundle.metadata["decision_threshold"])
    counterfactual_changes = (
        [
            {
                "feature": c["feature"],
                "from_value": float(c["from"]),
                "to_value": float(c["to"]),
                "resulting_probability": c["resulting_probability"],
            }
            for c in cf.changes
        ]
        if cf.changes
        else None
    )

    return build_explainability_report_html(
        applicant_id=application.customer_id or "(unspecified)",
        predicted_probability=proba,
        decision_threshold=bundle.metadata["decision_threshold"],
        shap_local=shap_local,
        plain_english_text=plain_text,
        lime_result=lime_result,
        counterfactual_changes=counterfactual_changes,
    )


# --- Tabular exports (CSV / Excel) -----------------------------------------
# Additive alongside the HTML reports above and the existing PDF applicant
# report (`/explain/report`) -- these three endpoints close the CSV/Excel
# gap for the platform's most naturally tabular datasets that previously had
# no export path at all (only paginated JSON). See `reporting/export.py`.

_EXPORT_MEDIA_TYPES = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def _tabular_response(rows: list[dict], *, fmt: str, filename_stem: str) -> Response:
    fmt = fmt.lower()
    if fmt not in _EXPORT_MEDIA_TYPES:
        fmt = "csv"
    body = rows_to_csv_bytes(rows) if fmt == "csv" else rows_to_excel_bytes(rows, sheet_name=filename_stem[:31])
    return Response(
        content=body,
        media_type=_EXPORT_MEDIA_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="{filename_stem}.{fmt}"'},
    )


@router.get("/export/predictions")
def export_predictions(
    format: str = "csv",
    limit: int = 1000,
    current_user=Depends(require_roles(*_REPORT_READ_ROLES)),
    scoring: ScoringService = Depends(get_scoring_service),
):
    """Downloadable prediction history -- the same records `GET
    /predict/history` returns as paginated JSON, flattened to a file an
    analyst can open directly in Excel or load into another BI tool."""
    rows = scoring.get_history(limit=min(limit, 5000))
    return _tabular_response(rows, fmt=format, filename_stem="prediction_history")


@router.get("/export/audit-logs")
def export_audit_logs(
    format: str = "csv",
    limit: int = 1000,
    current_user=Depends(require_roles(*roles_at_least("admin"), "auditor")),
    audit: AuditService = Depends(get_audit_service),
):
    """Downloadable audit-log export -- same RBAC as `GET /admin/audit-logs`
    (admin+ or auditor), since this is the same compliance data, just as a
    file instead of a paginated JSON page."""
    params = pagination_params(page=1, size=min(limit, 5000), sort_by=None, sort_order="desc")
    page = audit.list_logs(params)
    rows = [
        {
            "id": log.id,
            "user_id": log.user_id,
            "action": log.action,
            "resource": log.resource,
            "ip_address": log.ip_address,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in page.items
    ]
    return _tabular_response(rows, fmt=format, filename_stem="audit_logs")


@router.get("/export/model-comparison")
def export_model_comparison(
    format: str = "csv",
    current_user=Depends(require_roles(*_REPORT_READ_ROLES)),
    governance: ModelGovernanceService = Depends(get_model_governance_service),
):
    """Downloadable version of the `GET /reports/model-comparison` HTML
    report's underlying table."""
    try:
        rows = governance.get_comparison()
    except ModelNotFoundError:
        rows = []
    return _tabular_response(rows, fmt=format, filename_stem="model_comparison")


# --- AI Governance -----------------------------------------------------------
# Model Card / Data Card are documentation-style reports (see their module
# docstrings in reporting/model_card_report.py and
# reporting/data_card_report.py for what they do and do NOT claim to cover).
# The Compliance Dashboard is a pure aggregation of numbers every other
# subsystem already tracks (rule engine, workflow engine, model governance,
# data quality) -- no new business logic, no new persistence.


@router.get("/model-card", response_class=HTMLResponse)
def model_card_report(
    current_user=Depends(require_roles(*_REPORT_READ_ROLES)),
    bundle: ModelBundle = Depends(get_model_bundle),
    governance: ModelGovernanceService = Depends(get_model_governance_service),
):
    metrics = compute_evaluation_metrics(bundle)
    versions = governance.list_versions()
    is_champion = any(v["version"] == bundle.metadata["version"] and v.get("is_champion") for v in versions)
    return build_model_card_html(bundle.metadata, metrics, is_champion=is_champion)


@router.get("/data-card", response_class=HTMLResponse)
def data_card_report(
    current_user=Depends(require_roles(*_REPORT_READ_ROLES)),
    service: DataQualityService = Depends(get_data_quality_service),
):
    result = service.get_latest()
    return build_data_card_html(result)


@router.get("/compliance-dashboard")
def compliance_dashboard(
    current_user=Depends(require_roles(*_REPORT_READ_ROLES)),
    governance_service: GovernanceService = Depends(get_governance_service),
):
    """A single-call compliance/governance snapshot: current model
    governance status, business-rule inventory, recent workflow-execution
    outcomes, and the latest data-quality result -- the numbers a
    compliance/audit review would otherwise have to pull from four separate
    endpoints. See services/governance_service.py: every field here is read
    from data another subsystem already persists, computed once and shared
    with /copilot/summary/governance so the two can never drift."""
    return governance_service.compliance_dashboard()
