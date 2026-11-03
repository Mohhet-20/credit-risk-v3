"""
Authentication (JWT) and authorization (RBAC) primitives.

Design decision: access + refresh token pair rather than a single
long-lived token — access tokens are short-lived (30 min default) and used
on every request; refresh tokens are longer-lived (7 days) and only used to
mint new access tokens, so a leaked access token has a small blast radius.
Refresh tokens additionally carry a `ver` (token version) claim -- see
`services/auth_service.py::AuthService.refresh` -- so every refresh
ROTATES the token: the previous refresh token stops working the moment a
new one is issued, turning a stolen-but-not-yet-used refresh token into a
detectable replay rather than a standing backdoor. Passwords are hashed
with bcrypt via passlib (not rolled by hand). Role checks are implemented
as a FastAPI dependency factory (`require_role` / `require_roles`, see
`api/dependencies.py`) so routes declare their required role declaratively
rather than checking `current_user.role == "admin"` inline in every handler.

Design decision: `RolePower` is a 5-rung ORDINAL ladder
(viewer < loan_officer < risk_analyst < admin < super_admin) covering the
roles where "can do everything the tier below can" is actually true. The
two lateral specialist roles (`auditor`, `ml_engineer` -- see
`core/constants.py::UserRole`'s docstring) are deliberately NOT members of
this enum. `check_role()`'s existing fallback (`RolePower.__members__.get(actual_role, RolePower.viewer)`)
means a lateral role is treated as viewer-power for any ordinal
`require_role()` check -- correct default-deny behavior -- and specific
endpoints then opt them back in explicitly via `require_roles()`.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from enum import IntEnum

import pyotp
import jwt
from passlib.context import CryptContext

from configs.settings import get_settings
from core.exceptions import AuthenticationError, AuthorizationError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

MFA_CHALLENGE_TOKEN_EXPIRE_MINUTES = 5
MFA_ISSUER_NAME = "Credit Risk Intelligence Platform"
N_BACKUP_CODES = 8


class RolePower(IntEnum):
    """Ordinal role ranking so `require_role("risk_analyst")` also admits
    admins and super_admins. See module docstring for why `auditor` and
    `ml_engineer` are intentionally absent from this ladder."""

    viewer = 0
    loan_officer = 1
    risk_analyst = 2
    admin = 3
    super_admin = 4


def roles_at_least(minimum_role: str) -> set[str]:
    """Every ordinal role whose power is >= `minimum_role`'s -- used by
    `api/dependencies.py::require_roles()` call sites that want "the usual
    ordinal tier, PLUS this one lateral specialist role" without hand-listing
    every ordinal role name at each call site."""
    minimum_power = RolePower[minimum_role].value
    return {name for name, member in RolePower.__members__.items() if member.value >= minimum_power}


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, role: str, expires_minutes: int | None = None) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes or settings.access_token_expire_minutes)
    payload = {"sub": subject, "role": role, "exp": expire, "type": "access"}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(subject: str, token_version: int) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    payload = {"sub": subject, "exp": expire, "type": "refresh", "ver": token_version}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_mfa_challenge_token(subject: str) -> str:
    """A short-lived, narrowly-scoped token issued after a correct
    username/password when the account has MFA enabled -- it proves
    "password already verified" without being a usable access or refresh
    token itself (its `type` is neither `access` nor `refresh`, so it's
    rejected everywhere else `decode_token`'s callers check `type`)."""
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=MFA_CHALLENGE_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": subject, "exp": expire, "type": "mfa_challenge"}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise AuthenticationError(f"Invalid or expired token: {exc}") from exc


def check_role(required_role: str, actual_role: str) -> None:
    if required_role not in RolePower.__members__:
        # A programming error at the call site (e.g. `require_role("auditor")`
        # instead of `require_roles(...)`), not a request the caller can fix --
        # `auditor`/`ml_engineer` are deliberately not ordinal (see module
        # docstring), so raising a clear error here beats an opaque `KeyError`
        # surfacing as an unhandled 500 the next time someone adds a route.
        raise ValueError(
            f"check_role() called with non-ordinal required_role {required_role!r}. "
            f"Ordinal roles are {sorted(RolePower.__members__)}; for a lateral "
            "specialist role (e.g. 'auditor', 'ml_engineer'), use require_roles() instead."
        )
    required_power = RolePower[required_role].value
    actual_power = RolePower.__members__.get(actual_role, RolePower.viewer).value
    if actual_power < required_power:
        raise AuthorizationError(f"Role '{actual_role}' does not meet required role '{required_role}'.")


# --- MFA (TOTP) --------------------------------------------------------------
# Design decision: TOTP (RFC 6238), not SMS/email OTP -- no third-party
# delivery dependency (no SMS gateway, no SMTP integration needed for this
# to work end to end), works offline in any standard authenticator app
# (Google Authenticator, Authy, 1Password, etc.), and is the strongest
# widely-deployable second factor without hardware tokens (WebAuthn/FIDO2
# would be a reasonable future upgrade, not a downgrade, from here).


def generate_totp_secret() -> str:
    return pyotp.random_base32()


def build_totp_uri(secret: str, username: str) -> str:
    """`otpauth://` URI an authenticator app can consume directly (scan as
    a QR code, or paste — no QR-image generation dependency needed here)."""
    return pyotp.totp.TOTP(secret).provisioning_uri(name=username, issuer_name=MFA_ISSUER_NAME)


def verify_totp_code(secret: str, code: str) -> bool:
    """`valid_window=1` tolerates the code from one 30s step before/after
    now, which is standard practice for clock-skew tolerance between the
    server and the user's phone."""
    if not code or not code.isdigit():
        return False
    return pyotp.TOTP(secret).verify(code, valid_window=1)


def generate_backup_codes(n: int = N_BACKUP_CODES) -> list[str]:
    """Plaintext one-time backup codes, shown to the user exactly once at
    MFA-confirmation time -- callers are responsible for hashing each one
    (via `hash_password`, the same bcrypt context passwords use) before
    persisting, and never storing/logging the plaintext afterward."""
    return [f"{secrets.token_hex(4)}-{secrets.token_hex(4)}" for _ in range(n)]
