# Docker Compose Guide

Three compose files, each with a distinct job — this is Compose's standard multi-file pattern for
environment separation, not three unrelated configs:

| File | Role | How it's used |
|---|---|---|
| `docker-compose.yml` | Production-style base | `docker compose -f docker-compose.yml up -d`, or `./deployment/scripts/deploy_docker_compose.sh` |
| `docker-compose.override.yml` | Local dev overrides | Auto-merged by a bare `docker compose up -d` — no flag needed |
| `docker-compose.test.yml` | Containerized test run | `docker compose -f docker-compose.yml -f docker-compose.test.yml run --rm tests` |

**The override file merges automatically.** If `docker-compose.override.yml` sits next to
`docker-compose.yml`, a bare `docker compose up` picks it up with no `-f` flag — that's
Compose's documented default file-discovery behavior, not something wired up manually here. This is
exactly why `deploy_docker_compose.sh` explicitly pins `-f docker-compose.yml` throughout: a
"production deploy" script that silently picked up dev bind-mounts and `--reload` because someone
left the override file in place would be a subtle, easy-to-miss bug.

## Core stack (started by a bare `docker compose up`)

| Service | Image | Network(s) | Purpose |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | `backend` | Application database |
| `redis` | `redis:7-alpine` | `backend` | Shared cache (see `core/cache.py`) |
| `mlflow` | `python:3.10-slim` + pip-installed mlflow | `backend` | Experiment tracking / model registry |
| `api` | built from `Dockerfile.api` | `backend` + `frontend` | FastAPI backend |
| `frontend` | built from `Dockerfile.frontend` | `frontend` | Streamlit UI |

## Optional services (never started by default)

| Service | Profile | Start with |
|---|---|---|
| `ml-worker` | `ml` | `docker compose --profile ml run --rm ml-worker` |
| `adminer` | `tools` | `docker compose --profile tools up -d adminer` |

`ml-worker` is a run-to-completion training job, not a long-lived service — `run --rm` (not `up`)
is the correct invocation; it runs `pipelines.run_training --stage all` by default and exits.
`adminer` is a ~30MB single-container Postgres admin UI at `http://localhost:8080`, useful for local
debugging only. Both are gated behind Compose `profiles:` specifically so `docker compose up` with
no flags never starts them — a training job or an unauthenticated DB admin UI has no business
running just because someone ran the default command.

## Network isolation

Two bridge networks, not one:

- **`backend`** — Postgres, Redis, MLflow, the API, and the ML worker. Nothing here should be
  reachable from the Streamlit frontend directly.
- **`frontend`** — the Streamlit UI and the API. The API is the *only* service on both networks.

If the frontend container were ever compromised, it has no network path to Postgres or Redis at
all — only through the API's own authenticated HTTP surface. This mirrors the network segmentation
Kubernetes Services already give you for free per-namespace (`deployment/kubernetes/`); Compose
needs it declared explicitly via `networks:` on each service. `tests/infra/test_docker_compose_config.py`
asserts this split is actually in place (not just that both networks exist).

## Redis

`redis` backs `core/cache.py`'s `RedisCache` (see `docs/ARCHITECTURE.md` for the in-memory-vs-Redis
design decision). The `api` service is wired with `REDIS_URL=redis://redis:6379/0` in
`docker-compose.yml`, so running the full stack via Compose automatically exercises the Redis-backed
cache path rather than the in-memory default used by local `pytest`/`uvicorn --reload` runs. If
Redis is unreachable when the API starts, `get_cache()` logs a warning and falls back to the
in-memory backend rather than crashing — a cache is a performance optimization, not a hard
dependency.

## Resource limits

Every service declares `deploy.resources.limits`/`reservations` (cpus/memory). These apply as plain
container resource limits under `docker compose` v2 even outside Swarm mode (the Compose
Specification unified this behavior) — they aren't dead weight here. Defaults are conservative for
a laptop/single-host deployment; raise them via a further override file for a beefier host, or drop
them entirely if migrating to Kubernetes (where `resources.limits` on the Deployment spec is the
equivalent mechanism — see `deployment/kubernetes/05-api.yaml`).

## Running the test suite in a container

```bash
docker compose -f docker-compose.yml -f docker-compose.test.yml run --rm tests
```

This builds the exact `Dockerfile.api` image the real API would ship in and runs the full `pytest`
suite inside it against the real `postgres` service (not just the SQLite fixtures the unit/service
tests use) — verifying the actual deployable artifact, not a parallel test-only build that could
drift from it.

## Environment variables

`docker-compose.yml` reads `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB`/`JWT_SECRET_KEY` from
the shell environment or a `.env` file (auto-copied from `.env.example` by
`deploy_docker_compose.sh` on first run) with safe local-dev defaults if unset. Change all of them
before running anything beyond a laptop demo — see `docs/DEPLOYMENT.md`.

## What's NOT included in this compose stack

Disclosed explicitly rather than silently missing: a monitoring stack (Prometheus/Grafana) and a
message-queue-backed background job runner were both considered for this pass and deliberately left
out rather than added half-wired — there's no `/metrics` endpoint yet for Prometheus to scrape, and
no task queue consumer for a queue to feed. Adding either now would mean shipping a compose service
with nothing real behind it. See the infrastructure delivery summary in the project's chat history
for the full list of what this pass covered vs. deferred.
