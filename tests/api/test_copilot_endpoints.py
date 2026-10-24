"""API tests for the `/copilot` router -- RBAC boundaries and real end-to-end
responses wired to the actual explainability/monitoring/data-quality
services (via the `api_client` fixture's stubbed model bundle)."""

from __future__ import annotations

from tests.api.test_auth_and_predict import SAMPLE_APPLICATION, _register_and_login


def test_explain_prediction_requires_auth(api_client):
    resp = api_client.post("/api/v1/copilot/explain-prediction", json=SAMPLE_APPLICATION)
    assert resp.status_code == 401


def test_explain_prediction_returns_copilot_response(api_client):
    headers = _register_and_login(api_client, username="copilot1", role="loan_officer")
    resp = api_client.post("/api/v1/copilot/explain-prediction", json=SAMPLE_APPLICATION, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"headline", "narrative", "insights", "recommended_actions"}
    assert body["headline"]
    assert isinstance(body["insights"], list)
    assert isinstance(body["recommended_actions"], list)


def test_explain_prediction_viewer_role_forbidden(api_client):
    headers = _register_and_login(api_client, username="copilot2", role="viewer")
    resp = api_client.post("/api/v1/copilot/explain-prediction", json=SAMPLE_APPLICATION, headers=headers)
    assert resp.status_code == 403


def test_summary_model_performance_requires_risk_analyst_or_above(api_client):
    headers = _register_and_login(api_client, username="copilot3", role="loan_officer")
    resp = api_client.get("/api/v1/copilot/summary/model-performance", headers=headers)
    assert resp.status_code == 403

    headers_ra = _register_and_login(api_client, username="copilot3b", role="risk_analyst")
    resp2 = api_client.get("/api/v1/copilot/summary/model-performance", headers=headers_ra)
    assert resp2.status_code == 200
    body = resp2.json()
    assert set(body.keys()) == {"headline", "narrative", "insights", "recommended_actions"}


def test_summary_model_performance_admits_auditor_and_ml_engineer(api_client):
    for role in ("auditor", "ml_engineer"):
        headers = _register_and_login(api_client, username=f"copilot_{role}", role=role)
        resp = api_client.get("/api/v1/copilot/summary/model-performance", headers=headers)
        assert resp.status_code == 200, f"role={role}"


def test_summary_monitoring_returns_copilot_response(api_client):
    headers = _register_and_login(api_client, username="copilot4", role="risk_analyst")
    resp = api_client.get("/api/v1/copilot/summary/monitoring", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["headline"]


def test_summary_data_quality_404_when_no_report_yet(api_client):
    headers = _register_and_login(api_client, username="copilot5", role="risk_analyst")
    resp = api_client.get("/api/v1/copilot/summary/data-quality", headers=headers)
    assert resp.status_code == 404


def test_summary_data_quality_returns_copilot_response_after_run(api_client):
    headers = _register_and_login(api_client, username="copilot6", role="risk_analyst")
    run_resp = api_client.post("/api/v1/data-quality/run", headers=headers)
    assert run_resp.status_code == 200

    resp = api_client.get("/api/v1/copilot/summary/data-quality", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["headline"]


def test_executive_briefing_degrades_gracefully(api_client):
    """Even before a data-quality run has ever happened, the briefing still
    returns 200 with model + monitoring sections rather than erroring."""
    headers = _register_and_login(api_client, username="copilot7", role="risk_analyst")
    resp = api_client.get("/api/v1/copilot/executive-briefing", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "MODEL PERFORMANCE" in body["narrative"]
    assert "OPERATIONS" in body["narrative"]


def test_executive_briefing_forbidden_for_loan_officer(api_client):
    headers = _register_and_login(api_client, username="copilot8", role="loan_officer")
    resp = api_client.get("/api/v1/copilot/executive-briefing", headers=headers)
    assert resp.status_code == 403
