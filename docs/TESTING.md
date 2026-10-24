# Testing Guide

## Running the suite

```bash
make test           # full suite, verbose
make test-cov        # full suite + coverage report (htmlcov/, coverage.xml)
python3 -m pytest tests/unit tests/data tests/infra -q     # fast subset (~30s)
python3 -m pytest tests/api -q                              # API/HTTP layer
python3 -m pytest tests/integration -q                      # cross-module integration
```

All commands require `MLFLOW_ALLOW_FILE_STORE=true` in the environment (the `Makefile` targets already set it); this tells the MLflow tracking client it's safe to use a local file-store backend rather than requiring a configured tracking server, which is what the training/model-registry fixtures use.

## Directory layout

| Directory | What it covers |
|---|---|
| `tests/unit/` | Service-layer logic, security primitives, cache, repositories, reporting HTML generation, config validation — constructed directly against repository/service classes, no HTTP. |
| `tests/data/` | Data validation, profiling, outlier detection, drift, scoring. |
| `tests/api/` | Full HTTP-layer tests via FastAPI's `TestClient` — auth flows, RBAC boundaries, prediction endpoints, admin/health, data-quality, reports, rate limiting, MFA, lightweight performance smoke tests. |
| `tests/integration/` | Cross-module flows: feature pipeline end-to-end, model registry persistence. |
| `tests/infra/` | Docker/Compose configuration structure validation (no `docker` daemon required — parses YAML/Dockerfile text). |

## Fixtures (`tests/conftest.py`)

- `raw_dataset` / `trained_model_bundle`: a small (500-row) synthetic dataset and a fast `LogisticRegression` model, shared session-scoped across tests — correctness of the pipeline is what's under test, not model quality, so fixture speed matters more than realism.
- `db_session_factory`: an in-memory SQLite engine using SQLAlchemy's `StaticPool` so every session (including the one the API's request-handling code opens) shares the same underlying `:memory:` database.
- `api_client`: a FastAPI `TestClient` with `app.dependency_overrides` wired to the in-memory DB and the fixture model bundle, rate limiting disabled by default (re-enabled explicitly in `tests/api/test_rate_limiting.py`), and `client.db_session_factory` exposed so test helpers can seed data directly via the repository layer, bypassing HTTP (used for seeding privileged-role users now that public registration is locked to self-service roles — see `docs/SECURITY.md` §1.4).

## Testing privileged roles

Because `POST /auth/register` only accepts `viewer`/`loan_officer` (see `docs/SECURITY.md` §1.4), tests needing an `admin`/`super_admin`/`risk_analyst`/`auditor`/`ml_engineer` account don't go through HTTP registration. `tests/api/test_auth_and_predict.py::_register_and_login` is the shared helper: self-service roles register+login over real HTTP; privileged roles are seeded directly via `UserRepository.create()` against `client.db_session_factory()`, then a valid access token is hand-minted via `core.security.create_access_token()` — exactly mirroring how a real deployment bootstraps its first privileged accounts via `scripts/create_admin.py` rather than over the network.

## Known test-environment limitation: SQLite concurrency

`tests/api/test_performance.py::test_concurrent_predictions_all_succeed` serializes its DB-touching calls behind a `threading.Lock`. This is not a production concern — it's because `db_session_factory`'s `StaticPool` binds every session to one shared physical SQLite connection (necessary so the test and the API's request thread see the same in-memory DB), and a single `sqlite3` connection is not safe for two threads to execute statements on simultaneously even with `check_same_thread=False`. The documented production database is PostgreSQL (see `docker-compose.yml`'s `postgres` service), where every session gets its own real connection from a genuine pool — this limitation is specific to the SQLite-backed test fixture. See the comment above `_db_serialization_lock` in that file for the full explanation.

## Coverage philosophy

Every new endpoint or service method added in this pass has at least: one success-path test, one authorization-boundary test (wrong role gets 403, no auth gets 401), and one domain-exception test where applicable (e.g. duplicate registration, locked account, expired/replayed refresh token). RBAC boundary tests are organized in `tests/api/test_rbac_and_privileged_accounts.py` specifically so the full authorization matrix (§2.2 of `docs/SECURITY.md`) has one place documenting, in test form, what each role can and cannot do.

## CI

`.github/workflows/ci.yml` runs, on every push/PR: `lint` (ruff + black --check) → `test` (full pytest suite with coverage) → `build-images` (Docker image builds, push-only). A separate `dependency-audit` job runs `pip-audit` on every push/PR and weekly on a schedule, non-blocking (`continue-on-error: true`) — see `docs/SECURITY.md` §6 for why.
