"""API tests for the AI Governance endpoints (`/reports/model-card`,
`/reports/data-card`, `/reports/compliance-dashboard`) -- documentation-
style reports and a cross-subsystem aggregation, all built from data other
subsystems already track."""

from __future__ import annotations

from tests.api.test_auth_and_predict import _register_and_login


def test_model_card_renders_for_risk_analyst(api_client):
    headers = _register_and_login(api_client, username="gov_mc", role="risk_analyst")
    resp = api_client.get("/api/v1/reports/model-card", headers=headers)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Model Card" in resp.text
    assert "Ethical Considerations" in resp.text
    assert "Intended Use" in resp.text


def test_model_card_blocks_loan_officer(api_client):
    headers = _register_and_login(api_client, username="gov_mc_lo", role="loan_officer")
    resp = api_client.get("/api/v1/reports/model-card", headers=headers)
    assert resp.status_code == 403


def test_model_card_admits_auditor_and_ml_engineer(api_client):
    auditor_headers = _register_and_login(api_client, username="gov_mc_auditor", role="auditor")
    assert api_client.get("/api/v1/reports/model-card", headers=auditor_headers).status_code == 200

    ml_headers = _register_and_login(api_client, username="gov_mc_ml", role="ml_engineer")
    assert api_client.get("/api/v1/reports/model-card", headers=ml_headers).status_code == 200


def test_data_card_renders_gracefully_before_any_quality_run(api_client):
    headers = _register_and_login(api_client, username="gov_dc_empty", role="risk_analyst")
    resp = api_client.get("/api/v1/reports/data-card", headers=headers)
    assert resp.status_code == 200
    assert "Data Card" in resp.text
    assert "No data-quality report has been generated yet" in resp.text
    assert "synthetically generated" in resp.text.lower()


def test_data_card_shows_quality_snapshot_after_run(api_client):
    headers = _register_and_login(api_client, username="gov_dc_run", role="risk_analyst")
    run_resp = api_client.post("/api/v1/data-quality/run", headers=headers)
    assert run_resp.status_code == 200

    resp = api_client.get("/api/v1/reports/data-card", headers=headers)
    assert resp.status_code == 200
    assert "Quality Snapshot" in resp.text
    assert "Overall quality score" in resp.text


def test_compliance_dashboard_requires_authentication(api_client):
    resp = api_client.get("/api/v1/reports/compliance-dashboard")
    assert resp.status_code == 401


def test_compliance_dashboard_blocks_loan_officer(api_client):
    headers = _register_and_login(api_client, username="gov_dash_lo", role="loan_officer")
    resp = api_client.get("/api/v1/reports/compliance-dashboard", headers=headers)
    assert resp.status_code == 403


def test_compliance_dashboard_aggregates_governance_signals(api_client):
    ra_headers = _register_and_login(api_client, username="gov_dash_setup", role="risk_analyst")
    api_client.post(
        "/api/v1/rules",
        json={
            "name": "reject very poor credit",
            "rule_type": "rejection",
            "condition": {"field": "credit_score", "operator": "lt", "value": 400},
            "priority": 5,
        },
        headers=ra_headers,
    )

    lo_headers = _register_and_login(api_client, username="gov_dash_exec", role="loan_officer")
    from tests.api.test_auth_and_predict import SAMPLE_APPLICATION

    api_client.post("/api/v1/workflow/execute", json=SAMPLE_APPLICATION, headers=lo_headers)

    resp = api_client.get("/api/v1/reports/compliance-dashboard", headers=ra_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["business_rules"]["total_rules"] >= 1
    assert body["business_rules"]["active_rules"] >= 1
    assert "rejection" in body["business_rules"]["active_rules_by_type"]
    assert body["workflow"]["total"] >= 1
    assert "model_governance" in body
    assert "data_quality" in body
