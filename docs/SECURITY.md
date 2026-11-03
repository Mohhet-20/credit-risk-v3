# Security Guide

This document describes the Credit Risk Intelligence Platform's security model: authentication, authorization, secrets handling, transport/response hardening, input handling, dependency posture, and what is explicitly out of scope for this pass. It is written for engineers operating or extending the platform, and for anyone evaluating it for production readiness.

## 1. Authentication

### 1.1 Password-based login

- Passwords are hashed with **bcrypt** (via `passlib`), never stored or logged in plaintext.
- Password strength is enforced at registration/admin-creation time by `api/schemas.py::_validate_password_strength`: minimum 10 characters, at least one uppercase letter, one digit, and rejection of a small common-password blocklist (`_COMMON_PASSWORDS`). This runs as a Pydantic `field_validator` on both `UserCreate` (self-service) and `AdminUserCreate` (admin-issued), so there is exactly one policy, not two to keep in sync.
- Failed logins increment a per-user counter (`User.failed_login_attempts`). After `MAX_FAILED_LOGIN_ATTEMPTS` (see `core/constants.py`) consecutive failures, the account is locked (`User.locked_until` set) and `AuthService.login()` raises `AccountLockedError` (a subtype of `AuthenticationError`) even for the *correct* password until an admin calls `POST /admin/users/{id}/unlock` (`UserService.unlock_account`). This is a brute-force mitigation, not a rate limiter — see 1.5 for the complementary IP-based rate limit on `/auth/login`.

### 1.2 Tokens

- Access and refresh tokens are JWTs (`PyJWT`, HS256, secret from `JWT_SECRET_KEY`). `core/security.py::create_access_token`/`create_refresh_token` always pass an explicit `algorithm=` on encode and an explicit `algorithms=[...]` allow-list on decode — the API never accepts a token whose header claims a different algorithm than configured, which is what closes off the "algorithm confusion" JWT attack class regardless of the underlying library's own history (see §5).
- Access tokens are short-lived (`ACCESS_TOKEN_EXPIRE_MINUTES`, default 30). Refresh tokens are longer-lived (`REFRESH_TOKEN_EXPIRE_DAYS`, default 7) and are **single-use**: each refresh JWT embeds the user's current `token_version` (an integer on `User`); `AuthService.refresh()` rejects any refresh token whose embedded `ver` doesn't match the user's current `token_version` in the database, then increments `token_version` on every successful refresh. This means a stolen refresh token can be used exactly once before the legitimate holder's next refresh silently invalidates it (rotation), without needing a server-side token blacklist/store.
- `/auth/*` responses carry `Cache-Control: no-store` (see `api/middleware.py::SecurityHeadersMiddleware`) so tokens in JSON response bodies are never written to a shared HTTP cache or browser disk cache.
- This is a **bearer-token architecture**, not cookie-based sessions — "secure cookie flags" (`HttpOnly`/`Secure`/`SameSite`) are not applicable; the equivalent concern (token never persisted insecurely) is addressed by `no-store` plus the expectation that clients hold tokens in memory, not `localStorage`, on the web frontend (see `docs/DEVELOPER_GUIDE.md` for frontend integration notes).

### 1.3 Multi-factor authentication (TOTP)

- Real RFC 6238 TOTP, via `pyotp` — not a stub. Flow: `POST /auth/mfa/enroll` generates a secret and provisioning URI (scannable by any standard authenticator app); `POST /auth/mfa/confirm` verifies the user can produce a valid code before `User.mfa_enabled` flips true (proves possession before enabling, so a user can't lock themselves out with a secret they never actually set up); `POST /auth/mfa/disable` requires the current password plus either a TOTP code or a backup code.
- Backup codes (8 generated at enrollment, `secrets.token_hex`-based, bcrypt-hashed at rest, single-use — consumed and removed from the stored list on use) cover the lost-device case without a support-desk reset flow.
- Login with MFA enabled is two-step: `POST /auth/login` with correct password returns `{"mfa_required": true, "mfa_challenge_token": "..."}` instead of tokens; the challenge token is a short-lived (5 minute), narrowly-typed JWT (`mfa_challenge` type, rejected by every other endpoint) that must be presented to `POST /auth/mfa/verify-login` along with a TOTP or backup code to receive real access/refresh tokens. This bridges "password verified" and "second factor verified" without a mutable server-side session.
- MFA verify-login and the initial login are both rate-limited (see 1.5) — MFA does not disable brute-force protection on the first factor.

### 1.4 Registration lockdown (privilege escalation fix)

- **This pass closed a real privilege-escalation vulnerability**: the public `POST /auth/register` previously accepted an arbitrary `role` field, so anonymous callers could self-register as `admin`. `UserCreate.role` is now a Pydantic `Literal["viewer", "loan_officer"]` (rejected at the schema layer for any other value — a 422, not a 403, so it never reaches business logic), and `AuthService.register()` independently re-checks the requested role against `SELF_SERVICE_ROLES = frozenset({"viewer", "loan_officer"})` whenever `requesting_user is None` (i.e. the public endpoint). This is defense-in-depth, not redundancy: the schema check protects the public endpoint; the service-layer check protects every *other* caller of `register()` too, including tests and any future internal code path, from ever creating a privileged account without an authenticated `requesting_user`.
- Privileged accounts (`admin`, `super_admin`, `risk_analyst`, `auditor`, `ml_engineer`) can only be created two ways:
  1. `POST /admin/users` — requires an authenticated caller with `admin`-or-above role, using `AdminUserCreate` (the full 7-role `Literal`).
  2. `scripts/create_admin.py` — an out-of-band CLI (`python -m scripts.create_admin`) that talks directly to the repository layer, requiring the same OS/deployment-level trust as running a database migration. This is a deliberate choice over the common "first HTTP registration becomes admin" pattern, which is a known race-condition/attack-surface anti-pattern (whoever's registration request lands first over the network wins admin — including an attacker racing the real operator on a freshly deployed, still-empty instance).

### 1.5 Rate limiting & brute force

- Global default rate limit (`RATE_LIMIT_PER_MINUTE`, per client IP, via `slowapi`) applies to every endpoint by default.
- `/auth/login` and `/auth/mfa/verify-login` carry a stricter route-level limit (see `core/rate_limit.py`) — the two endpoints an attacker would target for credential/second-factor brute forcing.
- Account lockout (1.1) and rate limiting are complementary: rate limiting slows a distributed attack across many accounts; lockout stops a focused attack against one account even if it's slow enough to stay under the rate limit.

## 2. Authorization (RBAC)

### 2.1 The seven roles

`core/constants.py::UserRole` defines seven roles, split into two design categories:

**Ordinal roles** (`core/security.py::RolePower`, a strict linear hierarchy — each tier can do everything the tiers below it can): `viewer` < `loan_officer` < `risk_analyst` < `admin` < `super_admin`. `require_role("X")` and the `roles_at_least("X")` helper both use this ordering.

**Lateral roles** (deliberately *not* part of the ordinal ladder): `auditor` (read-heavy compliance access — audit logs, reports, monitoring — no write access to users, models, or predictions) and `ml_engineer` (model-governance writes — promote/rollback/reload-cache, data-quality runs — no user-management access). These don't fit "somewhere on the ladder": granting them ordinal power would either over-grant (an auditor able to promote models) or under-grant (an ml_engineer needing `admin` just to reload a model cache, which would also hand them user management). Endpoints that should admit a lateral role alongside the ordinal ladder use the `require_roles(*allowed)` dependency (an explicit allow-list) instead of `require_role("X")` (an ordinal "at least this tier" check). A lateral role's fallback under an ordinal `require_role()` check is `viewer`-equivalent power (see `check_role()`'s `RolePower.__members__.get(actual_role, RolePower.viewer)`), so a lateral role never accidentally inherits ordinal privileges it wasn't explicitly granted.

### 2.2 Authorization matrix (representative endpoints)

| Endpoint | Minimum access |
|---|---|
| `POST /predict`, `POST /predict/batch` | `loan_officer`+ |
| `GET /models/versions`, `GET /models/comparison` | any authenticated user (`viewer`+) — see 2.3 |
| `POST /models/promote`, `/rollback`, `/reload-cache` | `admin`+ or `ml_engineer` |
| `POST /data-quality/run` | `risk_analyst`+ or `ml_engineer` |
| `GET /data-quality/*`, `GET /reports/*` (model comparison, evaluation, drift, performance, data-quality) | `risk_analyst`+ or `auditor` or `ml_engineer` |
| `POST /reports/explainability` | `loan_officer`+ (business workflow surface, not a compliance artifact — intentionally not opened to `auditor`/`ml_engineer`) |
| `GET /admin/audit-logs` | `admin`+ or `auditor` |
| `POST /admin/users`, user activate/deactivate/unlock/role-change | `admin`+ |
| `GET /health/*` | unauthenticated (operational surface, not business data) |

### 2.3 Fixed unauthenticated-endpoint bug

`GET /models/versions` and `GET /models/comparison` previously had **no authentication dependency at all** — fully unauthenticated model metadata exposure. Both now require at least `viewer` (any authenticated user), the least-privileged fix that closes the gap without narrowing legitimate access.

### 2.4 Self-demotion / self-deactivation protection

`UserService.set_active_status()` and `.set_role()` both refuse to let an admin-tier user (`admin` or `super_admin`) deactivate their own account or demote themselves out of the admin tier — prevents a single admin from accidentally (or via a compromised session) locking out all admin access with no recovery path short of the out-of-band CLI.

## 3. Transport & response hardening

`api/middleware.py::SecurityHeadersMiddleware` adds to every response:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`
- `Content-Security-Policy: default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'` — relevant because `api/routers/reports.py` serves raw `HTMLResponse` pages a browser renders directly, not just JSON. `img-src data:` and `style-src 'unsafe-inline'` are scoped exceptions for the reports' self-contained base64-embedded matplotlib figures and inline `<style>` block (no external stylesheet/asset server by design — see `reporting/html_report.py`).
- `Strict-Transport-Security: max-age=63072000; includeSubDomains` — only outside `debug` mode (would be actively wrong advice for local HTTP dev).
- `Cache-Control: no-store` on `/auth/*` paths (see 1.2).

`CorrelationIdMiddleware` tags every request/response with an `X-Correlation-ID` (inbound-supplied or generated), propagated through every log line for the lifetime of the request via a `contextvars.ContextVar` — useful for incident correlation across the request → service → repository call stack, and across services if an upstream gateway supplies its own ID.

## 4. Input handling

### 4.1 Reflected XSS fix in HTML reports

`api/routers/reports.py` serves several `HTMLResponse` reports assembled by `reporting/*.py`. `build_explainability_report_html`'s `subtitle=f"Applicant: {applicant_id}"` embeds a value that traces back to `CreditApplicationRequest.customer_id` — a request body field. Before this pass, `reporting/html_report.py::render_report()` interpolated `title`/`subtitle` into the HTML template **unescaped**: a `customer_id` containing `<script>...</script>` would be reflected verbatim into the report and executed in the browser of whoever opened it. `render_report()` now calls `html.escape()` on both `title` and `subtitle` before interpolation. `body_html` is intentionally left un-escaped — it is assembled by each report generator from a fixed template plus already-escaped table data (`df_to_html_table` uses `.to_html(..., escape=True)`), so escaping it a second time would corrupt the markup. Regression tests: `tests/unit/test_reporting.py::test_explainability_report_escapes_malicious_applicant_id`, `::test_render_report_escapes_title_and_subtitle`.

### 4.2 Request validation

Every request body is a Pydantic model (`api/schemas.py`) with typed fields, explicit constraints (ranges, enums, string patterns) on the credit-application and admin-management schemas, and `extra="forbid"`-style strictness inherited from Pydantic v2 defaults for unrecognized fields on the models that define them explicitly. Malformed input is rejected as a 422 before it reaches any business logic — no request body field is ever interpolated into a raw SQL string (SQLAlchemy's ORM/query builder is used exclusively; there is no raw-SQL string formatting anywhere in `repositories/`).

### 4.3 File uploads

The API exposes **no file-upload endpoints** (`python-multipart` is a FastAPI/Starlette transitive dependency, not something any route in `api/routers/` currently uses) — file-upload-specific hardening (content-type sniffing, size limits, malware scanning, path traversal on stored filenames) is not applicable to the current surface. If a file-upload endpoint is added in the future, it should reuse the validation patterns in `api/schemas.py` and be reviewed against this section before merging.

### 4.4 Data masking

Passwords are never logged (structured logging in `core/logging.py` logs request path/method/status/duration/correlation-id, not request bodies). Prediction inputs persisted to the `predictions` table (`input_features` JSON column) are the applicant's financial data by design — this is the platform's core business data, not incidental PII leakage, and is protected by the RBAC boundaries in §2 rather than field-level masking. No credentials, tokens, or secrets are ever included in a log line or error response body — error handlers (`api/app.py`) return `{"detail": str(exc)}` from typed domain exceptions with deliberately-written user-facing messages, never a raw stack trace or exception `repr()`.

## 5. Secrets management

- All secrets (`JWT_SECRET_KEY`, `DATABASE_URL`, `REDIS_URL`) are sourced from environment variables via `configs/settings.py` (pydantic-settings), never hard-coded, following 12-factor conventions.
- **New this pass**: `Settings.validate_production_secrets()` / `validate_or_raise()` — a fail-fast check called from `api/app.py`'s startup `lifespan`. If `APP_ENV=production` and any of the following hold, the application **refuses to start**:
  - `JWT_SECRET_KEY` is still the shipped development default (`dev-only-insecure-secret-do-not-deploy`).
  - `JWT_SECRET_KEY` is shorter than 32 characters (inadequate HMAC signing strength).
  - `DEBUG=true` (suppresses HSTS; risks leaking stack traces via framework default error pages).
  - `DATABASE_URL` points at SQLite (does not support concurrent writers safely across multiple API replicas).
  
  All problems are collected and reported together (not just the first) so an operator fixing configuration doesn't have to trial-and-error through a restart loop. See `tests/unit/test_startup_validation.py`.
- Non-production environments (the default, `APP_ENV=development`) skip this check entirely — it is a "safe to run in prod" gate, not a general-purpose settings validator, so importing `Settings` in a test or local-dev context never raises.

## 6. Dependency security

- `requirements.txt` pins exact versions throughout.
- JWTs use `PyJWT` (2.15.1). `python-jose` was replaced because it and its `ecdsa` dependency have open advisories with no fixed release. Whatever the library, `core/security.py` always passes an explicit `algorithms=[settings.jwt_algorithm]` single-item allow-list to `jwt.decode()`, which is what blocks algorithm-confusion attacks such as CVE-2024-33663.
- `pyyaml` is pinned at `6.0.3`; the codebase contains no calls to `yaml.load()` or `yaml.unsafe_load()` anywhere (verified via repository-wide search) — only `yaml.safe_load`-equivalent usage patterns, so the historical PyYAML arbitrary-code-execution CVE class does not apply regardless of version.
- **Automated scanning**: `.github/workflows/ci.yml` now runs a `dependency-audit` job (`pip-audit` against `requirements.txt`) on every push/PR and on a weekly schedule (Monday 06:00 UTC, independent of any push, to catch newly-disclosed CVEs in already-merged dependencies). The job uploads a JSON report as a build artifact. It runs with `continue-on-error: true` deliberately: `pip-audit` requires outbound network access to PyPI's advisory data, which is unavailable in this development sandbox (confirmed: `pip-audit` hangs indefinitely here because the sandbox's network proxy returns `403` for the underlying advisory-database host) and may also be unavailable in firewalled corporate CI runners; a newly-disclosed transitive-dependency CVE should alert loudly, not silently block every merge for a team without a pre-agreed remediation SLA. **This means `pip-audit` has NOT been successfully executed against this exact dependency set in this environment** — the CI job is real and will run on a GitHub Actions runner with normal internet access, but that execution has not happened as part of this session. Tighten `continue-on-error` to a hard failure once your team has a documented dependency-CVE triage process.

## 7. What was explicitly scoped out this pass

Being direct about what this pass did **not** cover, so it isn't mistaken for a security sign-off on these areas:

- **True network-level load/stress testing.** `tests/api/test_performance.py` is an in-process `TestClient` smoke test (catches request-handler resource leaks and gross latency regressions cheaply in CI) — it is not a substitute for k6/Locust/Gatling against a deployed instance under sustained concurrent real-network traffic.
- **Email-based flows**: no email verification on registration, no password-reset-via-email flow. Password resets currently require an admin (`UserService`) or the account holder proving current-password + MFA (for MFA disable). Adding email verification/reset would need an email-delivery integration, which is out of scope for this pass.
- **Field-level PII encryption at rest.** Prediction inputs and user PII are protected by RBAC and standard database access controls, not column-level encryption. Appropriate for the current single-database deployment model; would need revisiting for stricter regulatory regimes (e.g. certain interpretations of GLBA/FCRA data-handling requirements) or a multi-tenant deployment.
- **Streamlit frontend accessibility beyond a contrast/labeling audit.** A WCAG AA color-contrast check was run against every custom color pair in `frontend/utils/theme.py` (all pass, several well above 4.5:1 — see the audit script referenced in this section's git history) and the app was confirmed to have no unlabeled images (`st.image` is unused) and no icon-only interactive controls lacking text labels. Deeper screen-reader support for the embedded Plotly charts is a Streamlit-platform-level limitation (Streamlit renders its own DOM; the app has no ARIA-attribute control over Plotly's internal chart markup) and is not fixable from application code alone.
- **A confirmed, successful `pip-audit` run in this exact environment** — see §6.
- **A full `black --format` pass across the pre-existing training/tests modules.** `black --check .` currently flags ~116 pre-existing files as non-conforming (a gap that predates this pass, not something introduced by it). A mass-reformat was deliberately not attempted in this pass given its size relative to the security/reliability work prioritized, and the risk of an unreviewed 116-file diff; `make format` runs it on demand for a maintainer who wants to close this gap deliberately.
