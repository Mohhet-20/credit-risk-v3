"""API tests for the AI Governance extension to the Copilot
(`/copilot/summary/governance`, and the governance section added to
`/copilot/executive-briefing`)."""

from __future__ import annotations

from tests.api.test_auth_and_predict import _register_and_login


def test_summary_governance_requires_authentication(api_client):
    resp = api_client.get("/api/v1/copilot/summary/governance")
    assert resp.status_code == 401


def test_summary_governance_blocks_loan_officer(api_client):
    headers = _register_and_login(api_client, username="cg_lo", role="loan_officer")
    resp = api_client.get("/api/v1/copilot/summary/governance", headers=headers)
    assert resp.status_code == 403


def test_summary_governance_returns_copilot_response_shape(api_client):
    headers = _register_and_login(api_client, username="cg_reader", role="risk_analyst")
    resp = api_client.get("/api/v1/copilot/summary/governance", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"headline", "narrative", "insights", "recommended_actions"}
    assert body["headline"]
    assert body["insights"]
    assert body["recommended_actions"]


def test_summary_governance_matches_compliance_dashboard_numbers(api_client):
    """The narrative copilot summary and the raw compliance dashboard must
    never disagree, since both are computed from the same
    GovernanceService.compliance_dashboard() call."""
    ra_headers = _register_and_login(api_client, username="cg_consistency_ra", role="risk_analyst")
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

    dashboard = api_client.get("/api/v1/reports/compliance-dashboard", headers=ra_headers).json()
    summary = api_client.get("/api/v1/copilot/summary/governance", headers=ra_headers).json()

    active_rules = dashboard["business_rules"]["active_rules"]
    assert str(active_rules) in summary["headline"]


def test_executive_briefing_includes_governance_section(api_client):
    headers = _register_and_login(api_client, username="cg_briefing", role="risk_analyst")
    resp = api_client.get("/api/v1/copilot/executive-briefing", headers=headers)
    assert resp.status_code == 200
    assert "GOVERNANCE" in resp.json()["narrative"]
