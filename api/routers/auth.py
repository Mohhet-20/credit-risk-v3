"""
Authentication endpoints: register, login, refresh, and MFA (TOTP)
enrollment / confirmation / disablement / login-verification.

Design decision: this router is a thin controller -- it parses the request,
calls `AuthService`, and shapes the response. Every business rule
(duplicate username check, self-service-role restriction, password
verification, active-account check, MFA branching, token issuance) lives in
`services/auth_service.py`; domain exceptions raised there
(`DuplicateResourceError`, `AuthenticationError`, `AuthorizationError`) are
translated to HTTP status codes by the handlers in `api/app.py`, so this
router doesn't need a single `try/except`.

Design decision: `POST /auth/login` always returns HTTP 200 with a
`LoginResponse` body, never a distinct status code for "MFA required" -- the
password WAS correct, so this isn't an error condition from the client's
perspective, just an incomplete login that `mfa_required=True` signals
explicitly in the body. See `docs/SECURITY.md` for the full MFA flow.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from api.dependencies import get_auth_service, get_current_user
from api.schemas import (
    LoginRequest,
    LoginResponse,
    MfaConfirmRequest,
    MfaConfirmResponse,
    MfaDisableRequest,
    MfaEnrollResponse,
    MfaVerifyLoginRequest,
    RefreshRequest,
    TokenResponse,
    UserCreate,
    UserOut,
)
from core.rate_limit import limiter
from database.models import User
from services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, auth_service: AuthService = Depends(get_auth_service)):
    return auth_service.register(
        username=payload.username,
        email=payload.email,
        password=payload.password,
        role=payload.role,
    )


@router.post("/login", response_model=LoginResponse)
@limiter.limit("10/minute")  # stricter than the global default -- login is a brute-force target
def login(payload: LoginRequest, request: Request, auth_service: AuthService = Depends(get_auth_service)):
    result = auth_service.login(
        username=payload.username,
        password=payload.password,
        ip_address=request.client.host if request.client else None,
    )
    if result.mfa_required:
        return LoginResponse(mfa_required=True, mfa_challenge_token=result.mfa_challenge_token)
    return LoginResponse(access_token=result.tokens.access_token, refresh_token=result.tokens.refresh_token)


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest, auth_service: AuthService = Depends(get_auth_service)):
    tokens = auth_service.refresh(refresh_token=payload.refresh_token)
    return TokenResponse(access_token=tokens.access_token, refresh_token=tokens.refresh_token)


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    """The authenticated caller's own profile -- lets a client render the
    signed-in user and role without decoding the JWT itself."""
    return current_user


@router.post("/mfa/enroll", response_model=MfaEnrollResponse)
def mfa_enroll(
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """Starts MFA enrollment: generates and stores a new TOTP secret
    (unconfirmed -- login is unaffected until `/auth/mfa/confirm` verifies a
    real code). Safe to call again to restart enrollment with a fresh secret."""
    result = auth_service.enroll_mfa(current_user.id)
    return MfaEnrollResponse(**result)


@router.post("/mfa/confirm", response_model=MfaConfirmResponse)
def mfa_confirm(
    payload: MfaConfirmRequest,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """Confirms enrollment with a real code from the user's authenticator
    app, enables MFA, and returns one-time backup codes -- shown exactly
    once here; only bcrypt hashes are persisted."""
    backup_codes = auth_service.confirm_mfa(current_user.id, payload.code)
    return MfaConfirmResponse(backup_codes=backup_codes)


@router.post("/mfa/disable", status_code=status.HTTP_204_NO_CONTENT)
def mfa_disable(
    payload: MfaDisableRequest,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    auth_service.disable_mfa(current_user.id, payload.password)


@router.post("/mfa/verify-login", response_model=TokenResponse)
@limiter.limit("10/minute")  # same brute-force-target reasoning as /auth/login
def mfa_verify_login(
    payload: MfaVerifyLoginRequest, request: Request, auth_service: AuthService = Depends(get_auth_service)
):
    tokens = auth_service.verify_mfa_login(mfa_challenge_token=payload.mfa_challenge_token, code=payload.code)
    return TokenResponse(access_token=tokens.access_token, refresh_token=tokens.refresh_token)
