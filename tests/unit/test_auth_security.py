"""
Unit tests (service-layer, no HTTP) for this pass's auth-hardening work:
password policy, refresh-token rotation, and TOTP-based MFA.
"""

from __future__ import annotations

import json

import pyotp
import pytest
from pydantic import ValidationError

from api.schemas import AdminUserCreate, UserCreate
from core.exceptions import AuthenticationError, AuthorizationError
from core.security import (
    build_totp_uri,
    generate_backup_codes,
    generate_totp_secret,
    hash_password,
    verify_totp_code,
)
from repositories.audit_log_repository import AuditLogRepository
from repositories.user_repository import UserRepository
from services.auth_service import AuthService

VALID_PASSWORD = "Sup3rSecure!Pass"


def _auth_service(db_session_factory) -> AuthService:
    db = db_session_factory()
    return AuthService(UserRepository(db), AuditLogRepository(db))


# --- Password policy ---------------------------------------------------------


def test_password_policy_rejects_missing_uppercase():
    with pytest.raises(ValidationError):
        UserCreate(username="pol1", email="pol1@example.com", password="lowercase1!", role="viewer")


def test_password_policy_rejects_missing_special_char():
    with pytest.raises(ValidationError):
        UserCreate(username="pol2", email="pol2@example.com", password="NoSpecial1", role="viewer")


def test_password_policy_rejects_missing_digit():
    with pytest.raises(ValidationError):
        UserCreate(username="pol3", email="pol3@example.com", password="NoDigits!!", role="viewer")


def test_password_policy_rejects_common_password():
    with pytest.raises(ValidationError):
        UserCreate(username="pol4", email="pol4@example.com", password="Password1", role="viewer")


def test_password_policy_accepts_compliant_password():
    user = UserCreate(username="pol5", email="pol5@example.com", password=VALID_PASSWORD, role="viewer")
    assert user.password == VALID_PASSWORD


def test_user_create_schema_rejects_privileged_role():
    with pytest.raises(ValidationError):
        UserCreate(username="pol6", email="pol6@example.com", password=VALID_PASSWORD, role="admin")


def test_admin_user_create_schema_accepts_any_of_the_seven_roles():
    for role in ("super_admin", "admin", "ml_engineer", "risk_analyst", "auditor", "loan_officer", "viewer"):
        payload = AdminUserCreate(username="pol7", email="pol7@example.com", password=VALID_PASSWORD, role=role)
        assert payload.role == role


# --- Refresh token rotation ---------------------------------------------------


def test_refresh_rotates_token_and_rejects_replay(db_session_factory):
    service = _auth_service(db_session_factory)
    service.register(username="rotor1", email="rotor1@example.com", password=VALID_PASSWORD, role="viewer")
    result = service.login(username="rotor1", password=VALID_PASSWORD)
    old_refresh = result.tokens.refresh_token

    rotated = service.refresh(refresh_token=old_refresh)
    assert rotated.refresh_token != old_refresh

    with pytest.raises(AuthenticationError):
        service.refresh(refresh_token=old_refresh)  # replay of the now-stale token


def test_refresh_chain_of_rotations_keeps_working(db_session_factory):
    service = _auth_service(db_session_factory)
    service.register(username="rotor2", email="rotor2@example.com", password=VALID_PASSWORD, role="viewer")
    result = service.login(username="rotor2", password=VALID_PASSWORD)

    token = result.tokens.refresh_token
    for _ in range(3):
        pair = service.refresh(refresh_token=token)
        token = pair.refresh_token
    assert token  # the chain never breaks as long as each new token is used exactly once


# --- MFA (TOTP) core helpers ---------------------------------------------------


def test_totp_uri_contains_issuer_and_username():
    secret = generate_totp_secret()
    uri = build_totp_uri(secret, "alice")
    assert uri.startswith("otpauth://totp/")
    assert "alice" in uri


def test_verify_totp_code_accepts_correct_code_rejects_wrong_code():
    secret = generate_totp_secret()
    assert verify_totp_code(secret, pyotp.TOTP(secret).now()) is True
    assert verify_totp_code(secret, "000000") is False


def test_verify_totp_code_rejects_non_numeric_input():
    secret = generate_totp_secret()
    assert verify_totp_code(secret, "not-a-code") is False
    assert verify_totp_code(secret, "") is False


def test_generate_backup_codes_returns_unique_codes():
    codes = generate_backup_codes(n=8)
    assert len(codes) == 8
    assert len(set(codes)) == 8


# --- MFA (TOTP) full service-layer flow ----------------------------------------


def test_mfa_enrollment_does_not_enable_mfa_until_confirmed(db_session_factory):
    db = db_session_factory()
    service = AuthService(UserRepository(db), AuditLogRepository(db))
    user = service.register(username="mfa1", email="mfa1@example.com", password=VALID_PASSWORD, role="viewer")

    service.enroll_mfa(user.id)
    result = service.login(username="mfa1", password=VALID_PASSWORD)
    assert result.mfa_required is False  # not enabled yet -- only enrolled


def test_mfa_confirm_with_wrong_code_raises(db_session_factory):
    db = db_session_factory()
    service = AuthService(UserRepository(db), AuditLogRepository(db))
    user = service.register(username="mfa2", email="mfa2@example.com", password=VALID_PASSWORD, role="viewer")
    service.enroll_mfa(user.id)

    with pytest.raises(AuthenticationError):
        service.confirm_mfa(user.id, "000000")


def test_mfa_full_flow_login_requires_challenge_then_totp(db_session_factory):
    db = db_session_factory()
    service = AuthService(UserRepository(db), AuditLogRepository(db))
    user = service.register(username="mfa3", email="mfa3@example.com", password=VALID_PASSWORD, role="viewer")

    enroll = service.enroll_mfa(user.id)
    totp = pyotp.TOTP(enroll["secret"])
    backup_codes = service.confirm_mfa(user.id, totp.now())
    assert len(backup_codes) == 8

    result = service.login(username="mfa3", password=VALID_PASSWORD)
    assert result.mfa_required is True
    assert result.tokens is None
    assert result.mfa_challenge_token

    with pytest.raises(AuthenticationError):
        service.verify_mfa_login(mfa_challenge_token=result.mfa_challenge_token, code="000000")

    tokens = service.verify_mfa_login(mfa_challenge_token=result.mfa_challenge_token, code=totp.now())
    assert tokens.access_token and tokens.refresh_token


def test_mfa_backup_code_is_single_use(db_session_factory):
    db = db_session_factory()
    service = AuthService(UserRepository(db), AuditLogRepository(db))
    user = service.register(username="mfa4", email="mfa4@example.com", password=VALID_PASSWORD, role="viewer")

    enroll = service.enroll_mfa(user.id)
    backup_codes = service.confirm_mfa(user.id, pyotp.TOTP(enroll["secret"]).now())
    code = backup_codes[0]

    challenge_1 = service.login(username="mfa4", password=VALID_PASSWORD).mfa_challenge_token
    service.verify_mfa_login(mfa_challenge_token=challenge_1, code=code)  # first use succeeds

    challenge_2 = service.login(username="mfa4", password=VALID_PASSWORD).mfa_challenge_token
    with pytest.raises(AuthenticationError):
        service.verify_mfa_login(mfa_challenge_token=challenge_2, code=code)  # second use rejected


def test_disable_mfa_requires_correct_password(db_session_factory):
    db = db_session_factory()
    service = AuthService(UserRepository(db), AuditLogRepository(db))
    user = service.register(username="mfa5", email="mfa5@example.com", password=VALID_PASSWORD, role="viewer")
    enroll = service.enroll_mfa(user.id)
    service.confirm_mfa(user.id, pyotp.TOTP(enroll["secret"]).now())

    with pytest.raises(AuthenticationError):
        service.disable_mfa(user.id, "wrong-password")

    service.disable_mfa(user.id, VALID_PASSWORD)
    result = service.login(username="mfa5", password=VALID_PASSWORD)
    assert result.mfa_required is False


def test_backup_codes_are_never_stored_in_plaintext(db_session_factory):
    db = db_session_factory()
    service = AuthService(UserRepository(db), AuditLogRepository(db))
    user = service.register(username="mfa6", email="mfa6@example.com", password=VALID_PASSWORD, role="viewer")
    enroll = service.enroll_mfa(user.id)
    backup_codes = service.confirm_mfa(user.id, pyotp.TOTP(enroll["secret"]).now())

    stored = UserRepository(db).get_by_id(user.id)
    hashed_codes = json.loads(stored.mfa_backup_codes)
    assert set(hashed_codes).isdisjoint(set(backup_codes))  # stored values are hashes, not the plaintext codes


# --- Registration lockdown (privilege-escalation fix) --------------------------


def test_register_rejects_privileged_role_for_anonymous_caller(db_session_factory):
    service = _auth_service(db_session_factory)
    with pytest.raises(AuthorizationError):
        service.register(username="escalator", email="esc@example.com", password=VALID_PASSWORD, role="admin")


def test_register_allows_privileged_role_when_requested_by_admin(db_session_factory):
    db = db_session_factory()
    user_repo = UserRepository(db)
    service = AuthService(user_repo, AuditLogRepository(db))
    admin = user_repo.create(
        username="realadmin",
        email="realadmin@example.com",
        hashed_password=hash_password(VALID_PASSWORD),
        role="admin",
    )

    created = service.register(
        username="newanalyst",
        email="na@example.com",
        password=VALID_PASSWORD,
        role="risk_analyst",
        requesting_user=admin,
    )
    assert created.role == "risk_analyst"
