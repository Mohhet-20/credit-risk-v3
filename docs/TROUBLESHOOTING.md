# Troubleshooting Guide

## Startup

**`InsecureConfigurationError: Refusing to start with an insecure production configuration`**
The app refused to boot because `APP_ENV=production` and one of the following is true (the error message lists every failing check, not just the first): `JWT_SECRET_KEY` is still the shipped dev default, `JWT_SECRET_KEY` is under 32 characters, `DEBUG=true`, or `DATABASE_URL` points at SQLite. Fix the listed environment variable(s) and restart. See `docs/SECURITY.md` §5. This check is skipped entirely when `APP_ENV` is not `production` (local dev/tests are unaffected).

**API container crash-loops on a fresh `docker compose up`**
`database/session.py::init_db_with_retry()` retries the initial DB connection up to 5 times with exponential backoff (0.5s → 8s) specifically to smooth over Postgres not yet accepting connections at the instant the API container starts. If it still fails after 5 attempts, the underlying `OperationalError` is raised and logged — check `docker compose logs postgres` for whether Postgres itself is healthy, and confirm `DATABASE_URL` matches the `postgres` service's actual host/port/credentials in your compose file.

**`unable to open database file` on SQLite**
`database/session.py::_ensure_sqlite_parent_dir()` auto-creates the parent directory of a file-based SQLite URL; if you still see this, check that the process has write permission to that path (common in a container running as a non-root user without a volume mount at the expected path).

## Authentication

**Login returns 200 with `mfa_required: true` and no tokens**
Expected behavior for an account with MFA enabled — see `docs/SECURITY.md` §1.3. Call `POST /auth/mfa/verify-login` with the returned `mfa_challenge_token` plus a current TOTP code or an unused backup code.

**"Account is locked" / repeated `AccountLockedError` even with the correct password**
The account hit `MAX_FAILED_LOGIN_ATTEMPTS` consecutive failed logins (see `core/constants.py`) and is locked until `User.locked_until` passes, or until an admin calls the unlock endpoint (`UserService.unlock_account`, exposed via `POST /admin/users/{id}/unlock`). This is intentional brute-force protection, not a bug.

**Refresh token rejected with `AuthenticationError` even though it hasn't expired**
Refresh tokens are single-use (rotation via `User.token_version` — see `docs/SECURITY.md` §1.2). If a refresh token is used twice (e.g. a client retries a failed request without checking whether the first attempt actually succeeded), the second use is correctly rejected. Clients should treat a refresh call as non-idempotent and never blindly retry it.

**`POST /auth/register` returns 422 for `role: "admin"`**
Intentional — see `docs/SECURITY.md` §1.4. Public registration only accepts `viewer`/`loan_officer`. Create privileged accounts via `POST /admin/users` (requires an authenticated admin) or `scripts/create_admin.py` (out-of-band bootstrap for the very first admin).

## Authorization

**403 on an endpoint you'd expect your role to access**
Check `docs/SECURITY.md` §2.2's authorization matrix. Two common gotchas: (1) `auditor` and `ml_engineer` are "lateral" roles, not part of the ordinal `viewer < loan_officer < risk_analyst < admin < super_admin` ladder — an endpoint gated with `require_role("risk_analyst")` (ordinal, "at least this tier") will NOT admit an `auditor` even though intuitively they're a senior compliance role; it needs `require_roles(*roles_at_least("risk_analyst"), "auditor")` (explicit allow-list) to include them. (2) A lateral role falls back to `viewer`-level power on any ordinal check it wasn't explicitly added to — this is deliberate (see §2.1), not a bug to route around by promoting the role's ordinal tier.

## Reports

**`GET /reports/data-quality` returns 404 "No data quality report has been generated yet"**
This endpoint renders the latest *persisted* report; it doesn't run a fresh check itself (that's a mutating operation, deliberately behind `POST /data-quality/run`, not a GET a browser tab or crawler could trigger accidentally). Run `POST /data-quality/run` first.

## Docker / Compose

See `docs/DOCKER.md` and `docs/DOCKER_COMPOSE.md` for image-build and multi-service orchestration troubleshooting (healthchecks, profiles, network isolation, environment-file layering).

## Tests

**`tests/api/test_performance.py::test_concurrent_predictions_all_succeed` fails locally after modifying `tests/conftest.py`**
If you removed or changed `_db_serialization_lock`'s usage, you likely reintroduced the SQLite single-connection concurrency issue described in `docs/TESTING.md`'s "Known test-environment limitation" section — this is a test-fixture artifact (`StaticPool` sharing one physical SQLite connection across threads), not a production bug.

**Full suite times out in a CI runner or sandboxed shell with a short command timeout**
Chunk the run by directory: `pytest tests/unit tests/data tests/infra`, then `pytest tests/integration`, then `pytest tests/api` (or smaller file groups within `tests/api` if still too slow) — the suite has grown large enough that a single invocation can exceed short per-command timeouts in constrained environments, independent of the tests themselves being fast (the full suite runs in well under a minute end to end).
