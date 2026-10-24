"""
Explainability endpoints: SHAP local explanation (+ plain English), a LIME
comparison method, counterfactual "what-if" scenarios, and downloadable
PDF reports.

Design decision: the SHAP and LIME explainers are built lazily per-request
against a background sample drawn from the training reference data (cached
at process scope via `lru_cache`, same pattern as the model bundle) rather
than persisting a fitted explainer object, since `TreeExplainer`/
`shap.Explainer`/`LimeTabularExplainer` construction is cheap relative to
model loading and keeps these routes stateless between requests.
"""

from __future__ import annotations

import uuid

import pandas as pd
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from api.dependencies import get_model_bundle, require_role
from api.schemas import (
    CounterfactualChange,
    CounterfactualResponse,
    CreditApplicationRequest,
    ExplanationContribution,
    ExplanationResponse,
    LimeContribution,
    LimeExplanationResponse,
)
from configs.settings import get_settings
from explainability import explainer_cache
from explainability.adverse_action import adverse_action_reasons
from explainability.counterfactuals import generate_counterfactual
from explainability.explanations import ModelExplainer
from explainability.lime_explainer import LimeExplainer
from explainability.narrative import explain_prediction_plain_english
from explainability.report_generator import generate_applicant_report
from models.registry import ModelBundle

router = APIRouter(prefix="/explain", tags=["explainability"])


def _build_explainer(bundle: ModelBundle) -> ModelExplainer:
    return explainer_cache.get_explainer(bundle)


def _build_lime_explainer(bundle: ModelBundle) -> LimeExplainer:
    # Same transformed feature space SHAP explains in (bundle.feature_pipeline
    # output), not the raw application schema -- so LIME's perturbations and
    # SHAP's contributions are directly comparable feature-for-feature.
    background_raw = explainer_cache.background_sample(bundle.metadata["version"])
    background_t = bundle.feature_pipeline.transform(background_raw)
    return LimeExplainer(background=background_t, feature_names=bundle.feature_pipeline.feature_names)


@router.post("", response_model=ExplanationResponse)
def explain_application(
    application: CreditApplicationRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
    current_user=Depends(require_role("loan_officer")),
):
    raw_df = pd.DataFrame([application.model_dump()])
    transformed = bundle.feature_pipeline.transform(raw_df)
    proba = float(bundle.model.predict_proba(transformed)[:, 1][0])

    explainer = _build_explainer(bundle)
    local = explainer.local_explanation(transformed)
    plain_text = explain_prediction_plain_english(local, proba, bundle.metadata["decision_threshold"])

    return ExplanationResponse(
        base_value=local["base_value"],
        predicted_value=local["predicted_value"],
        contributions=[ExplanationContribution(**c) for c in local["contributions"]],
        plain_english=plain_text,
        adverse_action_reasons=adverse_action_reasons(local["contributions"]),
    )


@router.post("/lime", response_model=LimeExplanationResponse)
def explain_lime(
    application: CreditApplicationRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
    current_user=Depends(require_role("loan_officer")),
):
    """LIME explanation for the same application `/explain` (SHAP) would
    score -- a comparison method, not a replacement. See
    `explainability/lime_explainer.py` for why the two can legitimately
    disagree on this kind of tabular data."""
    raw_df = pd.DataFrame([application.model_dump()])
    transformed = bundle.feature_pipeline.transform(raw_df)

    explainer = _build_lime_explainer(bundle)

    def predict_fn(df: pd.DataFrame):
        return bundle.model.predict_proba(df)[:, 1]

    result = explainer.explain_row(predict_fn, transformed)
    return LimeExplanationResponse(
        intercept=result["intercept"],
        local_prediction=result["local_prediction"],
        contributions=[LimeContribution(**c) for c in result["contributions"]],
    )


@router.post("/counterfactual", response_model=CounterfactualResponse)
def counterfactual_scenario(
    application: CreditApplicationRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
    current_user=Depends(require_role("loan_officer")),
):
    raw_df = pd.DataFrame([application.model_dump()])

    def predict_fn(transformed_df: pd.DataFrame):
        return bundle.model.predict_proba(transformed_df)[:, 1]

    result = generate_counterfactual(
        predict_fn,
        bundle.feature_pipeline,
        raw_df,
        threshold=bundle.metadata["decision_threshold"],
    )
    return CounterfactualResponse(
        achieved=result.achieved,
        original_probability=result.original_probability,
        final_probability=result.final_probability,
        n_steps=result.n_steps,
        changes=[
            CounterfactualChange(
                feature=c["feature"],
                from_value=float(c["from"]),
                to_value=float(c["to"]),
                resulting_probability=c["resulting_probability"],
            )
            for c in result.changes
        ],
    )


@router.post("/report")
def generate_report(
    application: CreditApplicationRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
    current_user=Depends(require_role("loan_officer")),
):
    raw_df = pd.DataFrame([application.model_dump()])
    transformed = bundle.feature_pipeline.transform(raw_df)
    proba = float(bundle.model.predict_proba(transformed)[:, 1][0])

    explainer = _build_explainer(bundle)
    local = explainer.local_explanation(transformed)
    plain_text = explain_prediction_plain_english(local, proba, bundle.metadata["decision_threshold"])

    def predict_fn(transformed_df: pd.DataFrame):
        return bundle.model.predict_proba(transformed_df)[:, 1]

    cf = generate_counterfactual(predict_fn, bundle.feature_pipeline, raw_df, bundle.metadata["decision_threshold"])

    settings = get_settings()
    report_id = str(uuid.uuid4())
    report_path = settings.reports_dir / f"applicant_{report_id}.pdf"
    generate_applicant_report(
        report_path,
        applicant_id=application.customer_id or report_id,
        predicted_probability=proba,
        decision_threshold=bundle.metadata["decision_threshold"],
        local_explanation=local,
        plain_english_text=plain_text,
        counterfactual_changes=cf.changes,
    )
    return FileResponse(str(report_path), media_type="application/pdf", filename=report_path.name)
