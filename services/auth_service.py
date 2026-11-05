"""
Authentication business logic: registration, login, MFA, token refresh.

Design decision: privileged roles can never be self-assigned through public
registration -- `register()` rejects any role outside `SELF_SERVICE_ROLES`
unless the call is made on behalf of an authenticated admin (`requesting_user`
set, only ever true from `POST /admin/users`, itself gated by
`require_role("admin")`). This closes what was previously a real
privilege-escalation hole: the public `/auth/register` endpoint used to
accept `role="admin"` from a completely anonymous caller. A fresh
deployment's first privileged account is created out-of-band via
`scripts/create_admin.py` (a server-side CLI, never reachable over HTTP) --
see `docs/SECURITY.md`.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

from configs.settings import get_settings
from core.constants import ACCOUNT_LOCKOUT_MINUTES, MAX_FAILED_LOGIN_ATTEMPTS
from core.exceptions import (
    AccountLockedError,
    AuthenticationError,
    AuthorizationError,
    DuplicateResourceError,
    ResourceNotFoundError,
)
from core.security import (
    build_totp_uri,
    create_access_token,
    create_mfa_challenge_token,
    create_refresh_token,
    decode_token,
    generate_backup_codes,
    generate_totp_secret,
    hash_password,
    verify_password,
    verify_totp_code,
)
from database.models import User
from repositories.audit_log_repository import AuditLogRepository
from repositories.user_repository import UserRepository

SELF_SERVICE_ROLES = frozenset({"viewer", "loan_officer"})
GUEST_ROLE = "risk_analyst"


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str


@dataclass
class LoginResult:
    """What `AuthService.login` returns: either a ready-to-use `TokenPair`
    (`mfa_required=False`), or a challenge token the caller must exchange at
    `verify_mfa_login` along with a TOTP/backup code (`mfa_required=True`,
    `tokens=None`)."""

    mfa_required: bool
    tokens: TokenPair | None = None
    mfa_challenge_token: str | None = None


def _as_aware_utc(value: datetime) -> datetime:
    """SQLite (the zero-setup local default, see database/session.py)
    doesn't actually persist timezone info even for a DateTime(timezone=True)
    column -- values round-trip back from the driver as naive. Postgres does
    preserve it. Normalizing here (assume naive == UTC, since every write
    path in this service uses datetime.now(timezone.utc)) keeps the lockout
    comparison correct on both backends instead of raising
    "TypeError: can't compare offset-naive and offset-aware datetimes" the
    moment a SQLite-backed User object has been reloaded from the DB."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class AuthService:
    def __init__(self, user_repo: UserRepository, audit_repo: AuditLogRepository):
        self.user_repo = user_repo
        self.audit_repo = audit_repo

    def register(
        self,
        *,
        username: str,
        email: str,
        password: str,
        role: str,
        requesting_user: User | None = None,
    ) -> User:
        if requesting_user is None and not get_settings().registration_enabled:
            raise ResourceNotFoundError("Self-service registration is turned off on this deployment.")
        if self.user_repo.get_by_username(username):
            raise DuplicateResourceError(f"Username '{username}' is already registered.")

        if requesting_user is None and role not in SELF_SERVICE_ROLES:
            raise AuthorizationError(
                f"Self-service registration cannot request role '{role}'. "
                "Ask an administrator to create privileged accounts."
            )

        user = self.user_repo.create(
            username=username,
            email=email,
            hashed_password=hash_password(password),
            role=role,
        )
        if requesting_user is None:
            self.audit_repo.log_action(user_id=user.id, action="user_registered", resource=f"user:{user.id}")
        else:
            self.audit_repo.log_action(
                user_id=requesting_user.id,
                action="user_created_by_admin",
                resource=f"user:{user.id}",
                details={"role": role},
            )
        return user

    def login(self, *, username: str, password: str, ip_address: str | None = None) -> LoginResult:
        user = self.user_repo.get_by_username(username)
        if user is None:
            # Same error as a wrong password below -- never reveal whether a
            # username exists via a different message/status code.
            raise AuthenticationError("Invalid username or password.")

        if user.locked_until is not None and _as_aware_utc(user.locked_until) > datetime.now(timezone.utc):
            self.audit_repo.log_action(
                user_id=user.id,
                action="login_blocked_locked",
                resource=f"user:{user.id}",
                ip_address=ip_address,
            )
            raise AccountLockedError(
                "Account is temporarily locked due to repeated failed login attempts. "
                f"Try again after {_as_aware_utc(user.locked_until).isoformat()}.",
            )

        if not verify_password(password, user.hashed_password):
            user = self.user_repo.record_failed_login(
                user,
                max_attempts=MAX_FAILED_LOGIN_ATTEMPTS,
                lockout_minutes=ACCOUNT_LOCKOUT_MINUTES,
            )
            self.audit_repo.log_action(
                user_id=user.id,
                action="login_failed",
                resource=f"user:{user.id}",
                ip_address=ip_address,
                details={"failed_attempts": user.failed_login_attempts},
            )
            if user.locked_until is not None:
                raise AccountLockedError(
                    f"Account locked after {MAX_FAILED_LOGIN_ATTEMPTS} failed attempts. "
                    f"Try again in {ACCOUNT_LOCKOUT_MINUTES} minutes.",
                )
            raise AuthenticationError("Invalid username or password.")

        if not user.is_active:
            raise AuthorizationError("Account is deactivated.")

        if user.mfa_enabled:
            # Password is correct, but a second factor is still required --
            # don't issue real tokens yet. Deliberately does NOT call
            # `record_successful_login` here: the login isn't complete until
            # `verify_mfa_login` succeeds.
            self.audit_repo.log_action(
                user_id=user.id,
                action="login_mfa_challenge_issued",
                resource=f"user:{user.id}",
                ip_address=ip_address,
            )
            return LoginResult(mfa_required=True, mfa_challenge_token=create_mfa_challenge_token(user.id))

        self.user_repo.record_successful_login(user)
        self.audit_repo.log_action(
            user_id=user.id,
            action="login",
            resource=f"user:{user.id}",
            ip_address=ip_address,
        )
        return LoginResult(mfa_required=False, tokens=self._issue_tokens(user))

    def refresh(self, *, refresh_token: str) -> TokenPair:
        token_data = decode_token(refresh_token)
        if token_data.get("type") != "refresh":
            raise AuthenticationError("Provided token is not a refresh token.")

        user = self.user_repo.get_by_id(token_data["sub"])
        if user is None or not user.is_active:
            raise AuthenticationError("User not found or inactive.")

        if token_data.get("ver") != user.token_version:
            # The presented refresh token predates the most recent rotation
            # -- either it was already consumed by a prior refresh, or
            # (more concerning) someone is replaying an old, potentially
            # stolen refresh token. Either way, reject it.
            self.audit_repo.log_action(
                user_id=user.id,
                action="refresh_token_reuse_detected",
                resource=f"user:{user.id}",
            )
            raise AuthenticationError("Refresh token has been rotated and is no longer valid.")

        user = self.user_repo.increment_token_version(user)  # rotate: this refresh token is now single-use
        self.audit_repo.log_action(user_id=user.id, action="token_refreshed", resource=f"user:{user.id}")
        return self._issue_tokens(user)

    def guest_session(self, *, ip_address: str | None = None) -> TokenPair:
        """Public demo: sign the visitor in as the shared guest account, a `risk_analyst` (score applications, see
        decline reasons and explanations, analytics, data quality, reports; no model promotion, no administration).
        Off unless GUEST_ACCESS_ENABLED. The account's password is random and never disclosed, and an account with
        more power than risk_analyst is never handed out as the guest."""
        settings = get_settings()
        if not settings.guest_access_enabled:
            raise ResourceNotFoundError("Guest access is not available on this deployment.")
        guest = self.user_repo.get_by_username(settings.guest_username)
        if guest is None:
            guest = self.user_repo.create(
                username=settings.guest_username,
                email=f"{settings.guest_username}@guest.example.com",
                hashed_password=hash_password(secrets.token_urlsafe(32)),
                role=GUEST_ROLE,
            )
        if guest.role != GUEST_ROLE or not guest.is_active:
            raise AuthorizationError("Guest access is misconfigured on this deployment.")
        self.audit_repo.log_action(
            user_id=guest.id,
            action="guest_session_started",
            resource=f"user:{guest.id}",
            ip_address=ip_address,
        )
        return self._issue_tokens(guest)

    def _issue_tokens(self, user: User) -> TokenPair:
        return TokenPair(
            access_token=create_access_token(user.id, user.role),
            refresh_token=create_refresh_token(user.id, user.token_version),
        )

    # --- MFA (TOTP) ----------------------------------------------------------
    def enroll_mfa(self, user_id: str) -> dict:
        """Generates a new TOTP secret and stores it (unconfirmed --
        `mfa_enabled` stays False, login behavior is unaffected until
        `confirm_mfa` verifies a real code)."""
        user = self.user_repo.get_by_id(user_id)
        if user is None:
            raise AuthenticationError("User not found.")
        secret = generate_totp_secret()
        self.user_repo.set_mfa_secret(user, secret=secret)
        self.audit_repo.log_action(user_id=user.id, action="mfa_enrollment_started", resource=f"user:{user.id}")
        return {"secret": secret, "otpauth_uri": build_totp_uri(secret, user.username)}

    def confirm_mfa(self, user_id: str, code: str) -> list[str]:
        """Verifies the first real code from the user's authenticator app,
        then enables MFA and generates backup codes. Returns the PLAINTEXT
        backup codes -- the only time they're ever available in plaintext;
        only bcrypt hashes are persisted."""
        user = self.user_repo.get_by_id(user_id)
        if user is None:
            raise AuthenticationError("User not found.")
        if not user.mfa_secret:
            raise AuthenticationError("No MFA enrollment in progress; call /auth/mfa/enroll first.")
        if not verify_totp_code(user.mfa_secret, code):
            self.audit_repo.log_action(user_id=user.id, action="mfa_confirmation_failed", resource=f"user:{user.id}")
            raise AuthenticationError("Invalid verification code.")

        backup_codes = generate_backup_codes()
        hashed = [hash_password(code_) for code_ in backup_codes]
        self.user_repo.enable_mfa(user, hashed_backup_codes_json=json.dumps(hashed))
        self.audit_repo.log_action(user_id=user.id, action="mfa_enabled", resource=f"user:{user.id}")
        return backup_codes

    def disable_mfa(self, user_id: str, password: str) -> None:
        user = self.user_repo.get_by_id(user_id)
        if user is None:
            raise AuthenticationError("User not found.")
        if not verify_password(password, user.hashed_password):
            raise AuthenticationError("Incorrect password.")
        self.user_repo.disable_mfa(user)
        self.audit_repo.log_action(user_id=user.id, action="mfa_disabled", resource=f"user:{user.id}")

    def verify_mfa_login(self, *, mfa_challenge_token: str, code: str) -> TokenPair:
        token_data = decode_token(mfa_challenge_token)
        if token_data.get("type") != "mfa_challenge":
            raise AuthenticationError("Provided token is not an MFA challenge token.")

        user = self.user_repo.get_by_id(token_data["sub"])
        if user is None or not user.is_active or not user.mfa_enabled:
            raise AuthenticationError("MFA challenge is no longer valid for this account.")

        if user.mfa_secret and verify_totp_code(user.mfa_secret, code):
            self.audit_repo.log_action(user_id=user.id, action="mfa_verification_success", resource=f"user:{user.id}")
        elif self._consume_backup_code(user, code):
            self.audit_repo.log_action(
                user_id=user.id,
                action="mfa_verification_success_via_backup_code",
                resource=f"user:{user.id}",
            )
        else:
            self.audit_repo.log_action(user_id=user.id, action="mfa_verification_failed", resource=f"user:{user.id}")
            raise AuthenticationError("Invalid verification code.")

        self.user_repo.record_successful_login(user)
        self.audit_repo.log_action(user_id=user.id, action="login", resource=f"user:{user.id}")
        return self._issue_tokens(user)

    def _consume_backup_code(self, user: User, code: str) -> bool:
        if not user.mfa_backup_codes:
            return False
        hashed_codes: list[str] = json.loads(user.mfa_backup_codes)
        for stored_hash in hashed_codes:
            if verify_password(code, stored_hash):
                hashed_codes.remove(stored_hash)  # single-use
                self.user_repo.set_backup_codes(user, hashed_backup_codes_json=json.dumps(hashed_codes))
                return True
        return False
