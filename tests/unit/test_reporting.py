"""
Unit tests for the reporting/ package's report generators, called directly
(no HTTP) against the same `trained_model_bundle` fixture used elsewhere.
"""

from __future__ import annotations

import pandas as pd

from explainability.explanations import ModelExplainer
from explainability.lime_explainer import LimeExplainer
from explainability.narrative import explain_prediction_plain_english
from reporting.drift_report import build_drift_report_html
from reporting.evaluation_report import build_evaluation_report_html, compute_evaluation_metrics
from reporting.explainability_report import build_explainability_report_html
from reporting.model_comparison_report import build_model_comparison_report_html
from reporting.performance_summary_report import build_performance_summary_html
from tests.api.test_auth_and_predict import SAMPLE_APPLICATION


def test_compute_evaluation_metrics_returns_full_metric_set(trained_model_bundle):
    metrics = compute_evaluation_metrics(trained_model_bundle, max_rows=200)
    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
        "calibration_curve",
        "lift_chart",
        "gain_chart",
        "threshold_analysis",
    ):
        assert key in metrics
    assert len(metrics["confusion_matrix"]) == 2 and len(metrics["confusion_matrix"][0]) == 2
    assert len(metrics["threshold_analysis"]) == 19
    assert len(metrics["lift_chart"]) == 10
    assert len(metrics["gain_chart"]) == 10


def test_evaluation_report_html_contains_key_sections(trained_model_bundle):
    metrics = compute_evaluation_metrics(trained_model_bundle, max_rows=200)
    html = build_evaluation_report_html(metrics, "logistic_regression", "test-v1")
    assert "<html" in html.lower()
    assert "Confusion Matrix" in html
    assert "Calibration" in html
    assert "Threshold Analysis" in html
    assert "data:image/png;base64," in html


def test_model_comparison_report_handles_missing_comparison_table():
    html = build_model_comparison_report_html([], [], None)
    assert "<html" in html.lower()
    assert "No model comparison table" in html


def test_model_comparison_report_renders_rows_and_versions():
    comparison_rows = [
        {"rank": 1, "model": "xgboost", "roc_auc": 0.91, "pr_auc": 0.82, "f1": 0.7},
        {"rank": 2, "model": "logistic_regression", "roc_auc": 0.85, "pr_auc": 0.74, "f1": 0.62},
    ]
    versions = [{"version": "v1", "model_name": "xgboost", "registered_at": "2024-01-01", "is_champion": True}]
    html = build_model_comparison_report_html(comparison_rows, versions, "v1")
    assert "xgboost" in html
    assert "Best candidate" in html
    assert "Registered Versions" in html


def test_drift_report_handles_error_case():
    html = build_drift_report_html({"error": "Not enough recent predictions."})
    assert "Not enough recent predictions" in html


def test_drift_report_renders_feature_drift():
    drift_result = {
        "overall_severity": "moderate",
        "feature_drift": [
            {"feature": "credit_score", "type": "numeric", "psi": 0.15, "severity": "moderate"},
            {"feature": "age", "type": "numeric", "psi": 0.02, "severity": "none"},
        ],
    }
    html = build_drift_report_html(drift_result)
    assert "MODERATE" in html
    assert "credit_score" in html


def test_performance_summary_handles_no_latency_samples():
    stats = {"n_predictions": 0, "avg_predicted_probability": None, "avg_latency_ms": None, "window_days": 30}
    latency_stats = {"n": 0}
    html = build_performance_summary_html(stats, latency_stats)
    assert "No latency samples" in html
    assert "Error-rate tracking is not yet instrumented" in html


def test_performance_summary_renders_latency_percentiles():
    stats = {"n_predictions": 120, "avg_predicted_probability": 0.18, "avg_latency_ms": 42.0, "window_days": 30}
    latency_stats = {"n": 120, "p50_ms": 30.0, "p95_ms": 80.0, "p99_ms": 110.0, "max_ms": 150.0}
    html = build_performance_summary_html(stats, latency_stats)
    assert "p95 latency" in html
    assert "80.0 ms" in html


def _application(payload: dict):
    from api.schemas import CreditApplicationRequest

    return CreditApplicationRequest(**payload)


def test_explainability_report_html_combines_shap_lime_and_counterfactual(trained_model_bundle):
    bundle = trained_model_bundle
    application = _application(SAMPLE_APPLICATION)
    raw_df = pd.DataFrame([application.model_dump()])
    transformed = bundle.feature_pipeline.transform(raw_df)
    proba = float(bundle.model.predict_proba(transformed)[:, 1][0])

    shap_explainer = ModelExplainer(model=bundle.model, feature_names=bundle.feature_pipeline.feature_names)
    shap_explainer.build(transformed)
    local = shap_explainer.local_explanation(transformed)
    plain_text = explain_prediction_plain_english(local, proba, bundle.metadata["decision_threshold"])

    lime_explainer = LimeExplainer(background=transformed, feature_names=bundle.feature_pipeline.feature_names)

    def predict_fn(df):
        return bundle.model.predict_proba(df)[:, 1]

    lime_result = lime_explainer.explain_row(predict_fn, transformed, num_features=5)

    html = build_explainability_report_html(
        applicant_id="TEST-001",
        predicted_probability=proba,
        decision_threshold=bundle.metadata["decision_threshold"],
        shap_local=local,
        plain_english_text=plain_text,
        lime_result=lime_result,
        counterfactual_changes=None,
    )
    assert "<html" in html.lower()
    assert "SHAP Local Explanation" in html
    assert "LIME Comparison" in html
    assert "TEST-001" in html


def test_explainability_report_escapes_malicious_applicant_id(trained_model_bundle):
    """Security regression test: `customer_id` traces back to a request body
    field (see `reporting/html_report.py::render_report`'s docstring on the
    reflected-XSS fix). A malicious customer_id must never appear as raw,
    executable markup in the rendered report."""
    bundle = trained_model_bundle
    application = _application(SAMPLE_APPLICATION)
    raw_df = pd.DataFrame([application.model_dump()])
    transformed = bundle.feature_pipeline.transform(raw_df)
    proba = float(bundle.model.predict_proba(transformed)[:, 1][0])

    shap_explainer = ModelExplainer(model=bundle.model, feature_names=bundle.feature_pipeline.feature_names)
    shap_explainer.build(transformed)
    local = shap_explainer.local_explanation(transformed)
    plain_text = explain_prediction_plain_english(local, proba, bundle.metadata["decision_threshold"])

    malicious_id = "<script>alert('xss')</script>"
    html = build_explainability_report_html(
        applicant_id=malicious_id,
        predicted_probability=proba,
        decision_threshold=bundle.metadata["decision_threshold"],
        shap_local=local,
        plain_english_text=plain_text,
        lime_result=None,
        counterfactual_changes=None,
    )
    assert "<script>alert('xss')</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_report_escapes_title_and_subtitle():
    from reporting.html_report import render_report

    html = render_report(
        title="<img src=x onerror=alert(1)>",
        subtitle="Applicant: <b>bold</b>",
        body_html="<p>trusted body content</p>",
    )
    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "&lt;b&gt;bold&lt;/b&gt;" in html
    # body_html is intentionally passed through un-escaped -- it's assembled
    # from trusted templates plus already-escaped table data.
    assert "<p>trusted body content</p>" in html
