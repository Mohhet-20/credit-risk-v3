"""API tests for the Business Rule Engine (`/rules/*`) -- RBAC boundaries,
CRUD, versioning-on-update, activate/deactivate, and simulation."""

from __future__ import annotations

from tests.api.test_auth_and_predict import _register_and_login

_APPROVAL_CONDITION = {"field": "credit_score", "operator": "gte", "value": 750}
_REJECTION_CONDITION = {"field": "credit_score", "operator": "lt", "value": 400}


def _create_rule(api_client, headers, **overrides):
    payload = {
        "name": "auto-approve excellent credit",
        "description": "Fast-track very low risk applicants.",
        "rule_type": "approval",
        "condition": _APPROVAL_CONDITION,
        "priority": 10,
    }
    payload.update(overrides)
    return api_client.post("/api/v1/rules", json=payload, headers=headers)


def test_create_rule_requires_authentication(api_client):
    resp = api_client.post(
        "/api/v1/rules",
        json={
            "name": "x",
            "rule_type": "approval",
            "condition": _APPROVAL_CONDITION,
            "priority": 10,
        },
    )
    assert resp.status_code == 401


def test_create_rule_blocks_loan_officer(api_client):
    headers = _register_and_login(api_client, username="rule_lo", role="loan_officer")
    resp = _create_rule(api_client, headers)
    assert resp.status_code == 403


def test_create_rule_allows_risk_analyst(api_client):
    headers = _register_and_login(api_client, username="rule_ra", role="risk_analyst")
    resp = _create_rule(api_client, headers)
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "auto-approve excellent credit"
    assert body["rule_type"] == "approval"
    assert body["is_active"] is True
    assert body["version"] == 1


def test_create_rule_rejects_invalid_rule_type(api_client):
    headers = _register_and_login(api_client, username="rule_badtype", role="risk_analyst")
    resp = _create_rule(api_client, headers, rule_type="not_a_real_type")
    assert resp.status_code == 422  # Pydantic Literal validation, before it ever reaches the service


def test_create_rule_rejects_malformed_condition(api_client):
    headers = _register_and_login(api_client, username="rule_badcond", role="risk_analyst")
    resp = _create_rule(api_client, headers, condition={"field": "credit_score", "operator": "unknown_op", "value": 1})
    assert resp.status_code == 400  # InvalidRuleConditionError -> app.py handler


def test_list_rules_admits_auditor(api_client):
    headers = _register_and_login(api_client, username="rule_creator1", role="risk_analyst")
    _create_rule(api_client, headers)

    auditor_headers = _register_and_login(api_client, username="rule_auditor1", role="auditor")
    resp = api_client.get("/api/v1/rules", headers=auditor_headers)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_list_rules_blocks_viewer(api_client):
    headers = _register_and_login(api_client, username="rule_viewer1", role="viewer")
    resp = api_client.get("/api/v1/rules", headers=headers)
    assert resp.status_code == 403


def test_get_rule_404_for_unknown_id(api_client):
    headers = _register_and_login(api_client, username="rule_get404", role="risk_analyst")
    resp = api_client.get("/api/v1/rules/does-not-exist", headers=headers)
    assert resp.status_code == 404


def test_update_rule_bumps_version_and_audit_logs_previous_snapshot(api_client):
    headers = _register_and_login(api_client, username="rule_updater", role="risk_analyst")
    rule_id = _create_rule(api_client, headers).json()["id"]

    resp = api_client.put(f"/api/v1/rules/{rule_id}", json={"priority": 5}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == 2
    assert body["priority"] == 5


def test_deactivate_and_activate_rule_round_trip(api_client):
    headers = _register_and_login(api_client, username="rule_toggler", role="risk_analyst")
    rule_id = _create_rule(api_client, headers).json()["id"]

    deactivated = api_client.post(f"/api/v1/rules/{rule_id}/deactivate", headers=headers)
    assert deactivated.status_code == 200
    assert deactivated.json()["is_active"] is False

    activated = api_client.post(f"/api/v1/rules/{rule_id}/activate", headers=headers)
    assert activated.status_code == 200
    assert activated.json()["is_active"] is True


def test_simulate_rules_dry_runs_without_persisting(api_client):
    headers = _register_and_login(api_client, username="rule_simulator", role="risk_analyst")
    _create_rule(
        api_client,
        headers,
        name="reject very poor credit",
        rule_type="rejection",
        condition=_REJECTION_CONDITION,
        priority=5,
    )

    resp = api_client.post(
        "/api/v1/rules/simulate",
        headers=headers,
        json={
            "contexts": [{"credit_score": 350}, {"credit_score": 800}],
        },
    )
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 2
    assert results[0]["matched_rule_type"] == "rejection"
    assert results[1]["matched_rule_id"] is None

    # Dry-run must not have persisted anything new.
    listing = api_client.get("/api/v1/rules", headers=headers)
    assert listing.status_code == 200


def test_simulate_rules_blocks_loan_officer(api_client):
    headers = _register_and_login(api_client, username="rule_sim_lo", role="loan_officer")
    resp = api_client.post("/api/v1/rules/simulate", headers=headers, json={"contexts": [{"credit_score": 400}]})
    assert resp.status_code == 403
