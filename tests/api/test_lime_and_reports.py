"""API tests for the LIME comparison endpoint and the new downloadable HTML
report endpoints (model comparison, evaluation, drift, performance summary,
per-applicant explainability)."""

from __future__ import annotations

from tests.api.test_auth_and_predict import SAMPLE_APPLICATION, _register_and_login


def test_lime_endpoint_returns_contributions(api_client):
    headers = _register_and_login(api_client, username="lime_tester")
    resp = api_client.post("/api/v1/explain/lime", json=SAMPLE_APPLICATION, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "lime"
    assert len(body["contributions"]) > 0
    assert all("rule" in c and "weight" in c for c in body["contributions"])


def test_lime_endpoint_requires_authentication(api_client):
    resp = api_client.post("/api/v1/explain/lime", json=SAMPLE_APPLICATION)
    assert resp.status_code == 401


def test_lime_endpoint_blocks_viewer_role(api_client):
    headers = _register_and_login(api_client, username="lime_viewer", role="viewer")
    resp = api_client.post("/api/v1/explain/lime", json=SAMPLE_APPLICATION, headers=headers)
    assert resp.status_code == 403


def test_evaluation_report_returns_html(api_client):
    headers = _register_and_login(api_client, username="eval_report_tester", role="risk_analyst")
    resp = api_client.get("/api/v1/reports/evaluation", headers=headers)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Model Evaluation Report" in resp.text
    assert "Confusion Matrix" in resp.text


def test_evaluation_report_blocks_loan_officer(api_client):
    headers = _register_and_login(api_client, username="eval_report_lo", role="loan_officer")
    resp = api_client.get("/api/v1/reports/evaluation", headers=headers)
    assert resp.status_code == 403


def test_model_comparison_report_returns_html(api_client):
    headers = _register_and_login(api_client, username="comparison_report_tester", role="risk_analyst")
    resp = api_client.get("/api/v1/reports/model-comparison", headers=headers)
    assert resp.status_code == 200
    assert "Model Comparison Report" in resp.text


def test_drift_report_returns_html(api_client):
    headers = _register_and_login(api_client, username="drift_report_tester", role="risk_analyst")
    resp = api_client.get("/api/v1/reports/drift", headers=headers)
    assert resp.status_code == 200
    assert "Drift Report" in resp.text


def test_performance_summary_report_returns_html(api_client):
    headers = _register_and_login(api_client, username="perf_report_tester", role="risk_analyst")
    resp = api_client.get("/api/v1/reports/performance-summary", headers=headers)
    assert resp.status_code == 200
    assert "Performance Summary" in resp.text


def test_explainability_report_returns_html_with_lime_section(api_client):
    headers = _register_and_login(api_client, username="explain_report_tester")
    resp = api_client.post("/api/v1/reports/explainability", json=SAMPLE_APPLICATION, headers=headers)
    assert resp.status_code == 200
    assert "Explainability Report" in resp.text
    assert "LIME Comparison" in resp.text
    assert "SHAP Local Explanation" in resp.text
