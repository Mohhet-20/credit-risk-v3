# Deployment Guide

All deployment paths build from three images defined at the repo root -- `Dockerfile.api` (FastAPI
backend), `Dockerfile.frontend` (Streamlit UI), and `Dockerfile.ml` (training / batch jobs). All
three are multi-stage builds, run as a non-root user, and share `requirements.txt` as the single
source of truth for dependency versions. See `docs/DOCKER.md` for the full image-level detail and
`docs/DOCKER_COMPOSE.md` for the compose-level detail (profiles, networking, resource limits) below.

## Local / single host — Docker Compose

```bash
./deployment/scripts/deploy_docker_compose.sh
```

This builds the api/frontend images and starts Postgres, Redis, MLflow, the API, and the frontend
via `docker-compose.yml` (production-style; the script explicitly ignores the dev-only
`docker-compose.override.yml`). On first run it copies `.env.example` to `.env` if missing — edit
`JWT_SECRET_KEY` before using this anywhere beyond a laptop demo. Once healthy:

- API: http://localhost:8000/docs
- Frontend: http://localhost:8501
- MLflow UI: http://localhost:5000

For local development with hot reload (bind-mounted source, `uvicorn --reload`), run
`docker compose up -d` directly instead of the script above — Compose auto-merges
`docker-compose.override.yml`. To run the training pipeline as a one-off container:
`docker compose --profile ml run --rm ml-worker`. To run the full pytest suite inside a container:
`docker compose -f docker-compose.yml -f docker-compose.test.yml run --rm tests`. See
`docs/DOCKER_COMPOSE.md` for the full breakdown.

## Kubernetes

Manifests live under `deployment/kubernetes/`, applied in numeric-prefix order:

| File | Purpose |
|---|---|
| `00-namespace.yaml` | Creates the `credit-risk` namespace |
| `01-configmap.yaml` | Non-secret env vars shared by api/frontend |
| `02-secret.yaml` | **Edit before applying** — placeholder JWT/DB secrets |
| `03-postgres.yaml` | Postgres Deployment + PVC + Service |
| `04-mlflow.yaml` | MLflow tracking server Deployment + PVC + Service |
| `05-api.yaml` | API Deployment (2 replicas + HPA) + Service |
| `06-frontend.yaml` | Frontend Deployment (2 replicas) + Service |
| `07-ingress.yaml` | TLS ingress (requires ingress-nginx + cert-manager) |

Before applying: push `credit-risk-api`/`credit-risk-frontend`/`credit-risk-mlflow` images to a
registry your cluster can pull from, update the `image:` fields in `05-api.yaml`/`06-frontend.yaml`/
`04-mlflow.yaml`, and replace every placeholder in `02-secret.yaml` (or better, provision it via an
External Secrets Operator instead of committing real values).

```bash
./deployment/scripts/deploy_kubernetes.sh
```

**Known gap, disclosed rather than silently stale:** these manifests predate the Redis cache
backend and the `Dockerfile.ml` image added in the latest infrastructure pass (this pass focused on
Docker/Compose; Kubernetes manifests weren't in scope for the update). There's no `redis` Deployment
here yet, and `core/cache.py::get_cache()` correctly falls back to the in-memory backend if
`REDIS_URL` is unset — nothing breaks by deploying as-is — but you won't get a shared,
multi-replica-safe cache in this cluster until a Redis Deployment/Service is added (mirror
`03-postgres.yaml`'s shape) and `01-configmap.yaml` is updated with a `REDIS_URL` pointing at it.

## Cloud-specific reference scripts

Each script is a working reference implementation, not a black box — read it before running it
against a real account, since it assumes supporting infrastructure (VPC, load balancer, managed
Postgres) already exists.

| Cloud | Script | Assumes |
|---|---|---|
| AWS | `deployment/scripts/deploy_aws_ecs.sh` | ECR repos + ECS cluster/services already provisioned |
| GCP | `deployment/scripts/deploy_gcp_cloud_run.sh` | Artifact Registry repo + Cloud SQL instance |
| Azure | `deployment/scripts/deploy_azure_container_apps.sh` | ACR + Container Apps environment |
| Render | `deployment/render.yaml` | Import as a Render Blueprint |
| Railway | `deployment/railway.json` | `railway up` from the repo root, or link in the dashboard |

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`/`master`: `ruff` + `black --check`,
then the full `pytest` suite with coverage (including the infra-validation tests under
`tests/infra/` that check the Dockerfiles and Compose files are well-formed without needing a Docker
daemon), then (on push only) a Docker build of all three images (api, frontend, ml) using GitHub
Actions' layer cache. It does not push images, scan them, or deploy — wire those in as separate
jobs once you have a registry and real cloud credentials to store as repo secrets.

## Environment variables

See `.env.example` for the full list. The most important ones to change before anything beyond
local development: `JWT_SECRET_KEY` (generate with
`python -c "import secrets; print(secrets.token_hex(32))"`), `DATABASE_URL`, and `APP_ENV=production`
(disables `debug` mode).
