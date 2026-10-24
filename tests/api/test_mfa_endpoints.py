"""API-level tests for the MFA (TOTP) endpoints: enroll, confirm, disable,
and the login -> mfa_required -> verify-login flow end to end."""

from __future__ import annotations

import pyotp

from tests.api.test_auth_and_predict import _register_and_login

VALID_PASSWORD = "Sup3rSecure!Pass"


def _register_plain_user(api_client, username: str) -> dict:
    api_client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "email": f"{username}@example.com",
            "password": VALID_PASSWORD,
            "role": "viewer",
        },
    )
    resp = api_client.post("/api/v1/auth/login", json={"username": username, "password": VALID_PASSWORD})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mfa_required"] is False
    return {"Authorization": f"Bearer {body['access_token']}"}


def test_login_without_mfa_returns_tokens_directly(api_client):
    headers = _register_plain_user(api_client, "mfa_api_plain1")
    assert "Authorization" in headers


def test_mfa_enroll_requires_authentication(api_client):
    resp = api_client.post("/api/v1/auth/mfa/enroll")
    assert resp.status_code == 401


def test_mfa_enroll_returns_secret_and_otpauth_uri(api_client):
    headers = _register_plain_user(api_client, "mfa_api_enroll1")
    resp = api_client.post("/api/v1/auth/mfa/enroll", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["secret"]
    assert body["otpauth_uri"].startswith("otpauth://")


def test_mfa_confirm_with_wrong_code_returns_401(api_client):
    headers = _register_plain_user(api_client, "mfa_api_confirm_wrong")
    api_client.post("/api/v1/auth/mfa/enroll", headers=headers)
    resp = api_client.post("/api/v1/auth/mfa/confirm", json={"code": "000000"}, headers=headers)
    assert resp.status_code == 401


def test_mfa_full_enroll_confirm_login_flow(api_client):
    username = "mfa_api_full1"
    headers = _register_plain_user(api_client, username)

    enroll_resp = api_client.post("/api/v1/auth/mfa/enroll", headers=headers)
    secret = enroll_resp.json()["secret"]
    totp = pyotp.TOTP(secret)

    confirm_resp = api_client.post("/api/v1/auth/mfa/confirm", json={"code": totp.now()}, headers=headers)
    assert confirm_resp.status_code == 200
    backup_codes = confirm_resp.json()["backup_codes"]
    assert len(backup_codes) == 8

    # Login now requires the second factor.
    login_resp = api_client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": VALID_PASSWORD},
    )
    assert login_resp.status_code == 200
    login_body = login_resp.json()
    assert login_body["mfa_required"] is True
    assert login_body["access_token"] is None
    challenge_token = login_body["mfa_challenge_token"]

    # Wrong code is rejected.
    bad_verify = api_client.post(
        "/api/v1/auth/mfa/verify-login",
        json={"mfa_challenge_token": challenge_token, "code": "000000"},
    )
    assert bad_verify.status_code == 401

    # Correct TOTP code completes login.
    good_verify = api_client.post(
        "/api/v1/auth/mfa/verify-login",
        json={"mfa_challenge_token": challenge_token, "code": totp.now()},
    )
    assert good_verify.status_code == 200
    assert good_verify.json()["access_token"]


def test_mfa_backup_code_completes_login(api_client):
    username = "mfa_api_backup1"
    headers = _register_plain_user(api_client, username)
    secret = api_client.post("/api/v1/auth/mfa/enroll", headers=headers).json()["secret"]
    totp = pyotp.TOTP(secret)
    backup_codes = api_client.post(
        "/api/v1/auth/mfa/confirm",
        json={"code": totp.now()},
        headers=headers,
    ).json()["backup_codes"]

    login_body = api_client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": VALID_PASSWORD},
    ).json()

    verify_resp = api_client.post(
        "/api/v1/auth/mfa/verify-login",
        json={"mfa_challenge_token": login_body["mfa_challenge_token"], "code": backup_codes[0]},
    )
    assert verify_resp.status_code == 200


def test_mfa_disable_requires_correct_password(api_client):
    username = "mfa_api_disable1"
    headers = _register_plain_user(api_client, username)
    secret = api_client.post("/api/v1/auth/mfa/enroll", headers=headers).json()["secret"]
    api_client.post("/api/v1/auth/mfa/confirm", json={"code": pyotp.TOTP(secret).now()}, headers=headers)

    bad_resp = api_client.post("/api/v1/auth/mfa/disable", json={"password": "wrong-password"}, headers=headers)
    assert bad_resp.status_code == 401

    good_resp = api_client.post("/api/v1/auth/mfa/disable", json={"password": VALID_PASSWORD}, headers=headers)
    assert good_resp.status_code == 204

    login_resp = api_client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": VALID_PASSWORD},
    )
    assert login_resp.json()["mfa_required"] is False


def test_mfa_endpoints_work_for_privileged_roles_too(api_client):
    """MFA is orthogonal to RBAC role -- an auditor or admin can enable it
    exactly the same way a viewer can."""
    headers = _register_and_login(api_client, username="mfa_api_auditor1", role="auditor")
    resp = api_client.post("/api/v1/auth/mfa/enroll", headers=headers)
    assert resp.status_code == 200
