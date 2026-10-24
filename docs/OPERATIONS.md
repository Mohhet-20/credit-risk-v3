# Operations Guide

Day-2 operations reference. For initial deployment, see `docs/DEPLOYMENT.md`, `docs/DOCKER.md`, and `docs/DOCKER_COMPOSE.md`.

## Health & readiness

| Endpoint | Purpose |
|---|---|
| `GET /health` | Bare liveness + champion-model summary (pre-existing, referenced by both Dockerfiles' `HEALTHCHECK` — do not change its shape). |
| `GET /health/liveness` | Always 200 if the process can handle a request at all, no I/O. A failure means the process is wedged; the orchestrator should restart the pod/container. |
| `GET /health/readiness` | 503 if the database is unreachable or no champion model is loaded. Signals "pull this instance out of the load balancer rotation without restarting it." |
| `GET /health/database` | Isolated DB connectivity check. |
| `GET /health/model` | Isolated champion-model-loaded check. |
| `GET /health/version` | App name/version/environment/Python version — useful for confirming what's actually deployed. |
| `GET /health/system` | Process uptime + platform info. |

Wire `/health/liveness` to your orchestrator's liveness probe and `/health/readiness` to its readiness probe — they express genuinely different failure semantics (restart vs. remove-from-service), which the single bare `/health` endpoint never could.

## Startup behavior

On boot (`api/app.py`'s `lifespan`), in order:

1. `settings.validate_or_raise()` — fails fast if this is a production deployment with an insecure configuration (see `docs/SECURITY.md` §5). **A production instance that fails to start here is refusing to run insecurely, not malfunctioning** — check the raised `InsecureConfigurationError`'s message, which lists every failing check.
2. `init_db_with_retry()` — up to 5 attempts with exponential backoff (0.5s → 8s) against `OperationalError`, smoothing over a database that isn't accepting connections yet at the exact moment the API container starts (a real race with `docker compose up` even with compose health-check dependencies).
3. Ready.

On shutdown, the cache backend's connection is released (`get_cache().close()` — a no-op for the in-memory backend, a real Redis connection-pool release otherwise) and the SQLAlchemy engine's connection pool is disposed (`engine.dispose()`), so a graceful restart (e.g. a Kubernetes rolling deploy sending `SIGTERM`) doesn't leak sockets.

## Configuration reference

See `.env.example` for the full list. The ones with operational consequences beyond "the app won't start":

- `JWT_SECRET_KEY` — see `docs/SECURITY.md` §5. Rotate by generating a new value (`openssl rand -hex 32`) and redeploying; this invalidates every outstanding access/refresh token immediately (no graceful rotation window currently — plan a maintenance window or brief re-auth prompt for connected clients).
- `REDIS_URL` — unset means the in-process `InMemoryTTLCache` (fine for a single replica; **not shared across replicas** — if you scale the API horizontally without setting this, each replica has its own independent cache, which is usually harmless for this platform's cached data (model-comparison rows, monitoring stats) but means cache invalidation on one replica doesn't propagate to others). Set it once you run more than one API replica.
- `RATE_LIMIT_PER_MINUTE` — global default; `/auth/login` and `/auth/mfa/verify-login` have their own stricter route-level limits regardless of this value (see `core/rate_limit.py`).
- `CORS_ORIGINS` — must include every origin the frontend is actually served from; misconfiguring this manifests as browser-console CORS errors, not a server-side error.

## Backup & recovery

- **Database**: standard PostgreSQL backup practices apply (`pg_dump`/continuous WAL archiving/managed-service snapshots) — this platform does not ship its own backup tooling, deliberately, since the right approach depends heavily on your PostgreSQL hosting choice (self-managed vs. RDS/Cloud SQL/etc.).
- **Model artifacts**: `models/artifacts/` (or the mounted equivalent in a container deployment) holds registered model versions; `mlruns/` holds the MLflow tracking history. Back these up together — the model registry's metadata references artifact paths, so restoring one without the other leaves a registry pointing at missing files.
- **Recovery drill**: promote/rollback (`POST /models/promote`, `POST /models/rollback`) already gives you a fast, tested path to revert a bad model deployment without a full restore — prefer it over a database restore for "the new model is behaving badly" incidents specifically.

## Rotating the admin bootstrap path

`scripts/create_admin.py` is the out-of-band CLI for creating the first privileged account (see `docs/SECURITY.md` §1.4) — run it against the deployment's actual database (`python -m scripts.create_admin`, prompts for username/email/password via `getpass`, never accepts a password as a CLI argument to avoid it landing in shell history). Subsequent privileged accounts should go through `POST /admin/users` once at least one admin exists.

## Monitoring what matters

- `GET /monitoring/drift`, `GET /reports/drift` — feature drift against the reference dataset.
- `GET /reports/performance-summary` — prediction latency percentiles and volume.
- `GET /admin/audit-logs` — every authentication event, role change, model promotion/rollback, and data-quality run is logged here with actor and correlation ID; this is the first place to look during an incident post-mortem.
- Structured JSON logs (`core/logging.py`) carry `correlation_id` on every line for the lifetime of a request — grep by correlation ID (from a client-reported `X-Correlation-ID` response header, or from an audit log entry) to reconstruct a single request's full call path across router → service → repository.

## Known operational gaps (see `docs/SECURITY.md` §7 for the full list)

- No automated backup scheduling shipped (deliberately deployment-specific — see above).
- No blue/green or canary deployment tooling beyond the model promote/rollback path — infrastructure-level deployment strategy is left to your orchestrator (Kubernetes rolling updates, a managed platform's native strategy, etc.).
- `pip-audit` dependency scanning runs in CI but has not been executed successfully in this development sandbox (network-restricted) — verify it runs clean on the first real CI run after merging.

## Database migrations (Alembic)

Schema changes are managed by Alembic (`database/migrations/`, `alembic.ini`), wired to
read the connection URL from `configs.settings.get_settings().database_url` at runtime
(see `database/migrations/env.py`) — the same `DATABASE_URL` every other part of the
platform reads, so migrations always target whichever database the app is actually
configured to use, in every environment, with nothing to keep in sync by hand.

```bash
make db-upgrade                       # apply all pending migrations
make db-downgrade                     # roll back the most recent migration
make db-current                       # show the currently applied revision
make db-history                       # show full migration history
make db-revision msg="add X column"   # autogenerate a migration from model changes
```

`init_db_with_retry()` (called from `api/app.py`'s startup lifespan) runs `alembic
upgrade head` in-process via `database/session.py::init_db()` — every environment
(local dev, Docker Compose, Kubernetes) applies pending migrations automatically on
API startup, against the same `DATABASE_URL` the rest of the platform reads. You do
not need a separate `make db-upgrade` deploy step; it's still available for manually
inspecting/advancing a database without starting the API (e.g. reviewing a pending
migration against a production database before the next rollout).
Autogenerated migrations are a starting point, not a guarantee — review the generated
file in `database/migrations/versions/` before applying it against a real database,
especially for any change that isn't a straightforward additive column/table.
