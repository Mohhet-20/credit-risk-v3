# Docker Guide

Three images, each purpose-built for a different runtime shape:

| Image | Dockerfile | Serves | Restart policy |
|---|---|---|---|
| API | `Dockerfile.api` | FastAPI backend, port 8000 | `unless-stopped` (long-running) |
| Frontend | `Dockerfile.frontend` | Streamlit UI, port 8501 | `unless-stopped` (long-running) |
| ML services | `Dockerfile.ml` | Training pipeline / batch jobs, no port | `"no"` (run-to-completion) |

Database (Postgres) and cache (Redis) use official upstream images (`postgres:16-alpine`,
`redis:7-alpine`) rather than custom Dockerfiles — there's no application code to bake into either.

## Why three separate images, not one

`Dockerfile.api` and `Dockerfile.frontend` serve live traffic and need to stay up: a health check,
a fast, predictable startup, and a restart policy that keeps them running. `Dockerfile.ml` runs a
training job (or any other batch script) to completion and exits — it has no HTTP server, so a
`HEALTHCHECK` directive would have nothing to probe; its "health signal" is its exit code, the same
way a Kubernetes `Job` differs from a `Deployment`. Shipping all three roles in one image would force
every training run to carry `uvicorn`/`fastapi`/`slowapi`'s footprint for no reason, and vice versa.
All three currently share `requirements.txt` and most of the codebase (a deliberate choice — see
`docs/ARCHITECTURE.md` — since maintaining separate per-image dependency files that can silently
drift out of sync isn't worth it at this project's scale); the separation that matters is the
runtime *role*, not the dependency graph.

## Multi-stage builds

Every custom Dockerfile is a two-stage build:

1. **`builder`** — installs everything from `requirements.txt` into a virtualenv at `/opt/venv`.
   `Dockerfile.api` and `Dockerfile.ml` additionally install `gcc`/`g++` here in case a future
   dependency bump ever needs to compile from source (today's pinned versions all ship prebuilt
   manylinux wheels, so this is headroom, not a current necessity).
2. **`runtime`** — a fresh `python:3.10-slim` base. Only `COPY --from=builder /opt/venv /opt/venv`
   crosses the stage boundary, plus the runtime-only OS packages each image actually needs
   (`libgomp1` for XGBoost/LightGBM's OpenMP code paths; `curl` to back a `HEALTHCHECK`).

The payoff: pip's download/build cache, `gcc`/`g++`, and the builder stage's `apt` package lists
never reach the image that ships. `tests/infra/test_dockerfiles.py` asserts this pattern is present
in all three Dockerfiles (via the `COPY --from=builder /opt/venv /opt/venv` line), not just that two
`FROM` lines exist.

## Non-root containers

Every custom image creates and switches to an unprivileged `appuser` (uid 1000) before the final
`CMD` — nothing runs as root inside any container. The `docker-compose.override.yml` dev bind-mount
(`./:/app`) works cleanly against this because uid 1000 is the conventional first non-root user ID
on most Linux dev hosts; if your host user has a different uid, either run
`docker compose exec api chown -R appuser /app` once, or don't rely on the dev override's bind mount.

## Health checks

`Dockerfile.api` and `Dockerfile.frontend` both declare a `HEALTHCHECK` hitting their own
`/health` / `/_stcore/health` endpoint, so `docker ps` and `docker inspect` report accurate
container health even outside Compose (e.g. a bare `docker run`). `docker-compose.yml` additionally
declares the *same* checks at the Compose level (not just relying on the image's built-in one) so
`depends_on: condition: service_healthy` has something to gate on, and so the check's
interval/timeout/retries can be tuned per environment without rebuilding the image.

## Building and running a single image directly

```bash
# API only, against whatever DATABASE_URL/MLFLOW_TRACKING_URI you export
docker build -f Dockerfile.api -t credit-risk-api .
docker run -p 8000:8000 --env-file .env credit-risk-api

# A single training run, writing artifacts to a mounted volume
docker build -f Dockerfile.ml -t credit-risk-ml .
docker run -v $(pwd)/models/artifacts:/app/models/artifacts credit-risk-ml \
  python3 -m pipelines.run_training --stage all
```

For the full stack (Postgres, Redis, MLflow, API, frontend), use Docker Compose instead — see
`docs/DOCKER_COMPOSE.md`.

## Image scanning and dependency scanning

Not wired into CI this pass (see the infrastructure delivery summary in the project's chat history
for the full list of what was scoped out). To scan an image manually before pushing it anywhere:

```bash
# Trivy (vulnerability scan of OS packages + Python deps baked into the image)
trivy image credit-risk-api

# pip-audit (Python dependency CVEs, doesn't require a built image)
pip install pip-audit && pip-audit -r requirements.txt
```
