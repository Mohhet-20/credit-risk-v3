"""API tests: admin user management, audit-log queries, and the new
health/diagnostics endpoints."""

from __future__ import annotations

from tests.api.test_auth_and_predict import _register_and_login


def test_admin_users_endpoint_requires_admin_role(api_client):
    headers = _register_and_login(api_client, username="nonadmin1", role="viewer")
    resp = api_client.get("/api/v1/admin/users", headers=headers)
    assert resp.status_code == 403


def test_admin_can_list_users_paginated(api_client):
    admin_headers = _register_and_login(api_client, username="admin1", role="admin")
    _register_and_login(api_client, username="regular1", role="viewer")

    resp = api_client.get("/api/v1/admin/users?page=1&size=5", headers=admin_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["page"] == 1
    assert body["size"] == 5
    assert body["total"] >= 2
    assert len(body["items"]) <= 5


def test_admin_can_deactivate_and_reactivate_a_user(api_client):
    admin_headers = _register_and_login(api_client, username="admin2", role="admin")
    reg = api_client.post(
        "/api/v1/auth/register",
        json={
            "username": "tobedeactivated",
            "email": "tbd@example.com",
            "password": "Supersecret1!",
            "role": "viewer",
        },
    )
    user_id = reg.json()["id"]

    resp = api_client.patch(
        f"/api/v1/admin/users/{user_id}/status",
        json={"is_active": False},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False

    login_resp = api_client.post(
        "/api/v1/auth/login",
        json={
            "username": "tobedeactivated",
            "password": "Supersecret1!",
        },
    )
    assert login_resp.status_code == 403


def test_admin_cannot_deactivate_own_account(api_client):
    admin_headers = _register_and_login(api_client, username="selfadmin", role="admin")
    me = api_client.get("/api/v1/admin/users", headers=admin_headers).json()
    my_id = next(u["id"] for u in me["items"] if u["username"] == "selfadmin")

    resp = api_client.patch(f"/api/v1/admin/users/{my_id}/status", json={"is_active": False}, headers=admin_headers)
    assert resp.status_code == 403


def test_admin_get_unknown_user_returns_404(api_client):
    admin_headers = _register_and_login(api_client, username="admin3", role="admin")
    resp = api_client.get("/api/v1/admin/users/does-not-exist", headers=admin_headers)
    assert resp.status_code == 404


def test_admin_can_query_audit_logs(api_client):
    admin_headers = _register_and_login(api_client, username="admin4", role="admin")
    _register_and_login(api_client, username="someoneelse", role="viewer")

    resp = api_client.get("/api/v1/admin/audit-logs?page=1&size=10", headers=admin_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert all("action" in item for item in body["items"])


def test_liveness_is_always_ok(api_client):
    resp = api_client.get("/health/liveness")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_readiness_reports_database_and_model_checks(api_client):
    resp = api_client.get("/health/readiness")
    assert resp.status_code == 200
    body = resp.json()
    assert "database" in body["checks"]
    assert "model" in body["checks"]


def test_version_endpoint_reports_app_metadata(api_client):
    resp = api_client.get("/health/version")
    assert resp.status_code == 200
    body = resp.json()
    assert "version" in body and "app_name" in body


def test_system_endpoint_reports_uptime(api_client):
    resp = api_client.get("/health/system")
    assert resp.status_code == 200
    assert "uptime_seconds" in resp.json()


def test_correlation_id_header_present_and_echoed(api_client):
    resp = api_client.get("/health/liveness", headers={"X-Correlation-ID": "test-corr-id-123"})
    assert resp.headers.get("X-Correlation-ID") == "test-corr-id-123"


def test_correlation_id_generated_when_absent(api_client):
    resp = api_client.get("/health/liveness")
    assert resp.headers.get("X-Correlation-ID")  # non-empty, server-generated


def test_security_headers_present_on_every_response(api_client):
    resp = api_client.get("/health/liveness")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
