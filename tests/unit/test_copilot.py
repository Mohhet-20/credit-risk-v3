"""
Unit tests for `explainability/copilot.py`'s pure narrative-generation
functions -- no FastAPI, no DB, plain dict-in/dict-out.
"""

from __future__ import annotations

from explainability.copilot import (
    explain_prediction_for_copilot,
    generate_executive_summary,
    summarize_data_quality,
    summarize_model_performance,
    summarize_monitoring,
)


def _every_response_has_the_uniform_shape(result: dict) -> None:
    assert set(result.keys()) == {"headline", "narrative", "insights", "recommended_actions"}
    assert isinstance(result["headline"], str) and result["headline"]
    assert isinstance(result["narrative"], str)
    assert isinstance(result["insights"], list)
    assert isinstance(result["recommended_actions"], list)


def test_explain_prediction_for_copilot_declined_with_counterfactual():
    contributions = [
        {"feature": "credit_score", "feature_value": 590, "shap_value": 0.12},
        {"feature": "debt_to_income_ratio", "feature_value": 45.0, "shap_value": 0.08},
    ]
    result = explain_prediction_for_copilot(
        plain_english_text="This application was declined...",
        predicted_probability=0.72,
        decision_threshold=0.5,
        shap_contributions=contributions,
        counterfactual_changes=[
            {"feature": "credit_score", "from_value": 590, "to_value": 650, "resulting_probability": 0.4}
        ],
    )
    _every_response_has_the_uniform_shape(result)
    assert "declined" in result["headline"]
    assert any("credit_score" in a for a in result["recommended_actions"])


def test_explain_prediction_for_copilot_approved_no_action_needed():
    result = explain_prediction_for_copilot(
        plain_english_text="This application was approved...",
        predicted_probability=0.1,
        decision_threshold=0.5,
        shap_contributions=[{"feature": "annual_income", "feature_value": 90000, "shap_value": -0.2}],
        counterfactual_changes=None,
    )
    _every_response_has_the_uniform_shape(result)
    assert "approved" in result["headline"]
    assert any("no action" in a.lower() for a in result["recommended_actions"])


def test_explain_prediction_for_copilot_declined_no_feasible_counterfactual():
    result = explain_prediction_for_copilot(
        plain_english_text="declined",
        predicted_probability=0.9,
        decision_threshold=0.5,
        shap_contributions=[{"feature": "credit_score", "feature_value": 500, "shap_value": 0.3}],
        counterfactual_changes=None,
    )
    assert any("manual" in a.lower() for a in result["recommended_actions"])


def test_summarize_model_performance_flags_low_precision():
    metrics = {
        "n_samples": 1000,
        "decision_threshold": 0.5,
        "accuracy": 0.7,
        "precision": 0.4,
        "recall": 0.9,
        "f1": 0.55,
        "roc_auc": 0.75,
        "pr_auc": 0.6,
    }
    result = summarize_model_performance(metrics, model_name="xgboost", model_version="v3")
    _every_response_has_the_uniform_shape(result)
    assert any("precision" in i.lower() for i in result["insights"])


def test_summarize_model_performance_healthy_metrics_no_concerns():
    metrics = {
        "n_samples": 1000,
        "decision_threshold": 0.5,
        "accuracy": 0.85,
        "precision": 0.8,
        "recall": 0.8,
        "f1": 0.8,
        "roc_auc": 0.88,
        "pr_auc": 0.8,
    }
    result = summarize_model_performance(metrics, model_name="xgboost", model_version="v3")
    assert any("strong" in i.lower() or "healthy" in i.lower() for i in result["insights"])


def test_summarize_monitoring_flags_high_drift():
    stats = {"n_predictions": 500, "avg_predicted_probability": 0.3, "window_days": 30}
    distribution = {"decision_counts": {"approved": 400, "rejected": 100}}
    drift = {"overall_severity": "high"}
    latency = {"n": 500, "p50_ms": 50, "p95_ms": 120, "p99_ms": 200, "max_ms": 300}
    result = summarize_monitoring(stats, distribution, drift, latency)
    _every_response_has_the_uniform_shape(result)
    assert any("drift" in i.lower() for i in result["insights"])
    assert any("retrain" in a.lower() or "investigate" in a.lower() for a in result["recommended_actions"])


def test_summarize_monitoring_no_predictions_flags_zero_traffic():
    stats = {"n_predictions": 0, "avg_predicted_probability": None, "window_days": 30}
    distribution = {"decision_counts": {}}
    drift = {"overall_severity": "stable"}
    latency = {"n": 0}
    result = summarize_monitoring(stats, distribution, drift, latency)
    assert any("no predictions" in i.lower() for i in result["insights"])


def test_summarize_data_quality_failed_report():
    report = {
        "passed": False,
        "score": {
            "overall_score": 45.0,
            "validity_score": 40,
            "completeness_score": 50,
            "uniqueness_score": 45,
            "n_validation_issues": 3,
            "freshness_hours": 1,
        },
    }
    result = summarize_data_quality(report)
    _every_response_has_the_uniform_shape(result)
    assert "FAILED" in result["headline"]
    assert any("critical" in i.lower() for i in result["insights"])
    assert any("do not promote" in a.lower() for a in result["recommended_actions"])


def test_summarize_data_quality_passed_report():
    report = {
        "passed": True,
        "score": {
            "overall_score": 95.0,
            "validity_score": 95,
            "completeness_score": 98,
            "uniqueness_score": 92,
            "n_validation_issues": 0,
            "freshness_hours": 1,
        },
    }
    result = summarize_data_quality(report)
    assert "PASSED" in result["headline"]
    assert any("no action" in a.lower() for a in result["recommended_actions"])


def test_generate_executive_summary_combines_all_three_sections():
    model_summary = summarize_model_performance(
        {
            "n_samples": 100,
            "decision_threshold": 0.5,
            "accuracy": 0.8,
            "precision": 0.8,
            "recall": 0.8,
            "f1": 0.8,
            "roc_auc": 0.85,
            "pr_auc": 0.8,
        },
        "xgboost",
        "v1",
    )
    monitoring_summary = summarize_monitoring(
        {"n_predictions": 10, "avg_predicted_probability": 0.2, "window_days": 30},
        {"decision_counts": {"approved": 10}},
        {"overall_severity": "low"},
        {"n": 0},
    )
    dq_summary = summarize_data_quality(
        {
            "passed": True,
            "score": {
                "overall_score": 90,
                "validity_score": 90,
                "completeness_score": 90,
                "uniqueness_score": 90,
                "n_validation_issues": 0,
                "freshness_hours": 1,
            },
        }
    )
    result = generate_executive_summary(model_summary, monitoring_summary, dq_summary)
    _every_response_has_the_uniform_shape(result)
    assert "MODEL PERFORMANCE" in result["narrative"]
    assert "OPERATIONS" in result["narrative"]
    assert "DATA QUALITY" in result["narrative"]


def test_generate_executive_summary_degrades_gracefully_with_no_data():
    result = generate_executive_summary(None, None, None)
    _every_response_has_the_uniform_shape(result)
    assert "no data available" in result["headline"].lower()


def test_generate_executive_summary_handles_partial_data():
    monitoring_summary = summarize_monitoring(
        {"n_predictions": 5, "avg_predicted_probability": 0.1, "window_days": 30},
        {"decision_counts": {}},
        {"overall_severity": "stable"},
        {"n": 0},
    )
    result = generate_executive_summary(None, monitoring_summary, None)
    assert "OPERATIONS" in result["narrative"]
    assert "MODEL PERFORMANCE" not in result["narrative"]
