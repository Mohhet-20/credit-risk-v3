"""
Bonus analytics endpoints: customer risk segmentation, application-level
anomaly detection, loan structure recommendations, and a feature-importance
timeline across registered model versions.

Design decision: segmentation/anomaly detection are fit fresh, per request,
against a reference sample of the training population (same pattern as the
SHAP background sample in `routers/explain.py`) rather than persisted as
global singletons -- see the design-decision docstrings in
`analytics.segmentation` / `analytics.anomaly_detection` for the tradeoff.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends

from analytics.anomaly_detection import AnomalyDetector
from analytics.feature_importance_timeline import compute_feature_importance_timeline
from analytics.loan_recommendation import recommend_loan_structures
from analytics.segmentation import CustomerSegmenter
from api.dependencies import get_model_bundle, get_model_registry, require_role
from api.schemas import CreditApplicationRequest
from configs.settings import get_settings
from models.registry import ModelBundle, ModelRegistry

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _reference_frame(n: int = 500) -> pd.DataFrame:
    settings = get_settings()
    df = pd.read_csv(settings.data_raw_dir / "credit_applications.csv").drop(columns=["default_flag"])
    return df.sample(min(len(df), n), random_state=42)


@router.post("/segment")
def segment_customer(
    application: CreditApplicationRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
    current_user=Depends(require_role("loan_officer")),
):
    raw_df = pd.DataFrame([application.model_dump()])
    reference_raw = _reference_frame()
    reference_transformed = bundle.feature_pipeline.transform(reference_raw)

    segmenter = CustomerSegmenter().fit(reference_transformed, reference_raw)
    profile = segmenter.predict(bundle.feature_pipeline.transform(raw_df))

    return {
        "segment_id": profile.segment_id,
        "label": profile.label,
        "segment_size_in_reference_sample": profile.size,
        "segment_avg_credit_score": round(profile.avg_credit_score, 1),
        "segment_avg_utilization": round(profile.avg_utilization, 3),
        "segment_avg_debt_to_income": round(profile.avg_debt_to_income, 1),
    }


@router.post("/anomaly")
def detect_anomaly(
    application: CreditApplicationRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
    current_user=Depends(require_role("loan_officer")),
):
    raw_df = pd.DataFrame([application.model_dump()])
    reference_raw = _reference_frame()
    reference_transformed = bundle.feature_pipeline.transform(reference_raw)

    detector = AnomalyDetector().fit(reference_transformed)
    result = detector.score(bundle.feature_pipeline.transform(raw_df))

    return {
        "is_anomaly": result.is_anomaly,
        "anomaly_score": round(result.anomaly_score, 4),
        "top_deviant_features": result.top_deviant_features,
    }


@router.post("/recommend")
def recommend_loan(
    application: CreditApplicationRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
    current_user=Depends(require_role("loan_officer")),
):
    raw_df = pd.DataFrame([application.model_dump()])

    def predict_fn(transformed_df: pd.DataFrame):
        return bundle.model.predict_proba(transformed_df)[:, 1]

    options = recommend_loan_structures(
        predict_fn,
        bundle.feature_pipeline,
        raw_df,
        bundle.metadata["decision_threshold"],
    )
    return {
        "recommendations": [
            {
                "loan_amount": o.loan_amount,
                "loan_term_months": o.loan_term_months,
                "predicted_probability": round(o.predicted_probability, 4),
                "estimated_monthly_payment": o.estimated_monthly_payment,
                "approved": o.approved,
            }
            for o in options
        ],
        "n_options_found": len(options),
    }


@router.get("/feature-importance-timeline")
def feature_importance_timeline(
    registry: ModelRegistry = Depends(get_model_registry),
    current_user=Depends(require_role("risk_analyst")),
):
    return {"timeline": compute_feature_importance_timeline(registry)}
