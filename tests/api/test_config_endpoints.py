"""API tests for the Enterprise Configuration Center (`/config/*`) -- default
fallback behavior, RBAC (read: risk_analyst+/auditor/ml_engineer, write:
admin+ only), and the audit-logged upsert path."""

from __future__ import annotations

from tests.api.test_auth_and_predict import _register_and_login


def test_list_config_requires_authentication(api_client):
    resp = api_client.get("/api/v1/config")
    assert resp.status_code == 401


def test_list_config_blocks_loan_officer(api_client):
    headers = _register_and_login(api_client, username="cfg_lo", role="loan_officer")
    resp = api_client.get("/api/v1/config", headers=headers)
    assert resp.status_code == 403


def test_list_config_returns_defaults_before_any_override(api_client):
    headers = _register_and_login(api_client, username="cfg_reader", role="risk_analyst")
    resp = api_client.get("/api/v1/config", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    keys = {c["key"] for c in body}
    assert "workflow.fraud_check_enabled" in keys
    assert "feature_flags.rule_engine_enabled" in keys
    fraud_flag = next(c for c in body if c["key"] == "workflow.fraud_check_enabled")
    assert fraud_flag["value"] is False
    assert fraud_flag["is_default"] is True


def test_list_config_admits_auditor_and_ml_engineer(api_client):
    auditor_headers = _register_and_login(api_client, username="cfg_auditor", role="auditor")
    assert api_client.get("/api/v1/config", headers=auditor_headers).status_code == 200

    ml_headers = _register_and_login(api_client, username="cfg_ml", role="ml_engineer")
    assert api_client.get("/api/v1/config", headers=ml_headers).status_code == 200


def test_get_single_config_value_falls_back_to_default(api_client):
    headers = _register_and_login(api_client, username="cfg_single", role="risk_analyst")
    resp = api_client.get("/api/v1/config/notifications.enabled", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"key": "notifications.enabled", "value": True}


def test_get_single_config_value_unknown_key_returns_none(api_client):
    headers = _register_and_login(api_client, username="cfg_unknown", role="risk_analyst")
    resp = api_client.get("/api/v1/config/totally.made.up.key", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["value"] is None


def test_set_config_value_blocks_risk_analyst(api_client):
    """Reading config is risk_analyst+, but WRITING is admin-only -- these
    keys directly control automated decisioning behavior."""
    headers = _register_and_login(api_client, username="cfg_writer_ra", role="risk_analyst")
    resp = api_client.put(
        "/api/v1/config/workflow.fraud_check_enabled",
        headers=headers,
        json={
            "value": True,
            "category": "workflow",
        },
    )
    assert resp.status_code == 403


def test_set_config_value_allows_admin_and_persists(api_client):
    headers = _register_and_login(api_client, username="cfg_admin", role="admin")
    resp = api_client.put(
        "/api/v1/config/workflow.fraud_check_enabled",
        headers=headers,
        json={
            "value": True,
            "category": "workflow",
            "description": "Enabled for this test.",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["value"] is True
    assert body["is_default"] is False

    followup = api_client.get("/api/v1/config/workflow.fraud_check_enabled", headers=headers)
    assert followup.json()["value"] is True


def test_list_config_filters_by_category(api_client):
    headers = _register_and_login(api_client, username="cfg_category", role="risk_analyst")
    resp = api_client.get("/api/v1/config?category=notifications", headers=headers)
    assert resp.status_code == 200
    assert all(c["category"] == "notifications" for c in resp.json())
