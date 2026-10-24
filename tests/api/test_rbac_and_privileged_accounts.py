"""
API-level tests for this pass's RBAC expansion: the 7-role set, lateral
roles (auditor, ml_engineer) admitted alongside ordinal tiers via
`require_roles()`, the registration lockdown (privilege-escalation fix),
the new admin-only user-creation endpoint, and the previously-unauthenticated
model endpoints now requiring at least `viewer`.
"""

from __future__ import annotations

from tests.api.test_auth_and_predict import SAMPLE_APPLICATION, _register_and_login

VALID_PASSWORD = "Sup3rSecure!Pass"


# --- Registration lockdown ----------------------------------------------------


def test_public_register_rejects_admin_role(api_client):
    resp = api_client.post(
        "/api/v1/auth/register",
        json={
            "username": "wannabe_admin",
            "email": "wa@example.com",
            "password": VALID_PASSWORD,
            "role": "admin",
        },
    )
    assert resp.status_code == 422  # rejected at the Pydantic schema layer (UserCreate's Literal)


def test_public_register_rejects_risk_analyst_role(api_client):
    resp = api_client.post(
        "/api/v1/auth/register",
        json={
            "username": "wannabe_analyst",
            "email": "wra@example.com",
            "password": VALID_PASSWORD,
            "role": "risk_analyst",
        },
    )
    assert resp.status_code == 422


def test_public_register_allows_self_service_roles(api_client):
    for role in ("viewer", "loan_officer"):
        resp = api_client.post(
            "/api/v1/auth/register",
            json={
                "username": f"selfservice_{role}",
                "email": f"ss_{role}@example.com",
                "password": VALID_PASSWORD,
                "role": role,
            },
        )
        assert resp.status_code == 201
        assert resp.json()["role"] == role


# --- Admin-only user creation --------------------------------------------------


def test_admin_can_create_privileged_user(api_client):
    admin_headers = _register_and_login(api_client, username="creator_admin", role="admin")
    resp = api_client.post(
        "/api/v1/admin/users",
        json={
            "username": "created_analyst",
            "email": "ca@example.com",
            "password": VALID_PASSWORD,
            "role": "risk_analyst",
        },
        headers=admin_headers,
    )
    assert resp.status_code == 201
    assert resp.json()["role"] == "risk_analyst"


def test_non_admin_cannot_create_privileged_user(api_client):
    headers = _register_and_login(api_client, username="regular_loan_officer", role="loan_officer")
    resp = api_client.post(
        "/api/v1/admin/users",
        json={"username": "sneaky", "email": "sneaky@example.com", "password": VALID_PASSWORD, "role": "admin"},
        headers=headers,
    )
    assert resp.status_code == 403


# --- Fixed unauthenticated model endpoints -------------------------------------


def test_model_versions_requires_authentication(api_client):
    resp = api_client.get("/api/v1/models/versions")
    assert resp.status_code == 401


def test_model_comparison_requires_authentication(api_client):
    resp = api_client.get("/api/v1/models/comparison")
    assert resp.status_code == 401


def test_model_versions_accessible_to_any_authenticated_role(api_client):
    headers = _register_and_login(api_client, username="any_viewer", role="viewer")
    resp = api_client.get("/api/v1/models/versions", headers=headers)
    assert resp.status_code == 200


# --- Lateral role: auditor gets read access to audit logs, reports, drift, dq --


def test_auditor_can_read_audit_logs(api_client):
    headers = _register_and_login(api_client, username="auditor1", role="auditor")
    resp = api_client.get("/api/v1/admin/audit-logs", headers=headers)
    assert resp.status_code == 200


def test_auditor_cannot_manage_users(api_client):
    headers = _register_and_login(api_client, username="auditor2", role="auditor")
    resp = api_client.get("/api/v1/admin/users", headers=headers)
    assert resp.status_code == 403


def test_auditor_can_read_data_quality_history_but_cannot_trigger_run(api_client):
    headers = _register_and_login(api_client, username="auditor3", role="auditor")
    read_resp = api_client.get("/api/v1/data-quality/history", headers=headers)
    assert read_resp.status_code == 200

    write_resp = api_client.post("/api/v1/data-quality/run", headers=headers)
    assert write_resp.status_code == 403


def test_auditor_can_read_reports(api_client):
    headers = _register_and_login(api_client, username="auditor4", role="auditor")
    resp = api_client.get("/api/v1/reports/model-comparison", headers=headers)
    assert resp.status_code == 200


def test_auditor_can_read_drift(api_client):
    headers = _register_and_login(api_client, username="auditor5", role="auditor")
    resp = api_client.get("/api/v1/monitoring/drift", headers=headers)
    assert resp.status_code == 200


def test_auditor_cannot_score_predictions(api_client):
    headers = _register_and_login(api_client, username="auditor6", role="auditor")
    resp = api_client.post("/api/v1/predict", json=SAMPLE_APPLICATION, headers=headers)
    assert resp.status_code == 403


# --- Lateral role: ml_engineer gets model-write + data-quality-run access -----


def test_ml_engineer_can_trigger_data_quality_run(api_client):
    headers = _register_and_login(api_client, username="mleng1", role="ml_engineer")
    resp = api_client.post("/api/v1/data-quality/run", headers=headers)
    assert resp.status_code == 200


def test_ml_engineer_can_reach_model_write_endpoints(api_client):
    # `/models/reload-cache` shares the same `_MODEL_WRITE_ROLES` RBAC gate
    # as promote/rollback but has no dependency on the fake test registry
    # implementing a full write path -- it just busts the in-process model
    # cache, so this cleanly isolates "did the role check admit ml_engineer"
    # from "does the stub registry support this write."
    headers = _register_and_login(api_client, username="mleng2", role="ml_engineer")
    resp = api_client.post("/api/v1/models/reload-cache", headers=headers)
    assert resp.status_code == 204


def test_loan_officer_cannot_reach_model_write_endpoints(api_client):
    headers = _register_and_login(api_client, username="officer_no_ml", role="loan_officer")
    resp = api_client.post("/api/v1/models/reload-cache", headers=headers)
    assert resp.status_code == 403


def test_ml_engineer_cannot_manage_users(api_client):
    headers = _register_and_login(api_client, username="mleng3", role="ml_engineer")
    resp = api_client.get("/api/v1/admin/users", headers=headers)
    assert resp.status_code == 403


# --- super_admin sits above admin in the ordinal hierarchy ---------------------


def test_super_admin_can_access_admin_only_endpoints(api_client):
    headers = _register_and_login(api_client, username="superadmin1", role="super_admin")
    resp = api_client.get("/api/v1/admin/users", headers=headers)
    assert resp.status_code == 200


def test_super_admin_can_read_audit_logs(api_client):
    headers = _register_and_login(api_client, username="superadmin2", role="super_admin")
    resp = api_client.get("/api/v1/admin/audit-logs", headers=headers)
    assert resp.status_code == 200


# --- Self-demotion protection now covers super_admin too -----------------------


def test_super_admin_cannot_demote_self_out_of_admin_tier(api_client):
    admin_headers = _register_and_login(api_client, username="superadmin3", role="super_admin")
    me = api_client.get(
        "/api/v1/admin/users",
        headers=admin_headers,
    ).json()["items"]
    my_id = next(u["id"] for u in me if u["username"] == "superadmin3")

    resp = api_client.patch(
        f"/api/v1/admin/users/{my_id}/role",
        json={"role": "viewer"},
        headers=admin_headers,
    )
    assert resp.status_code == 403
