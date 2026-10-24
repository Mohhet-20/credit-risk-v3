#!/usr/bin/env bash
# Local / single-host deployment via Docker Compose -- PRODUCTION-style
# stack (postgres, redis, mlflow, api, frontend). Explicitly pins to
# `-f docker-compose.yml` only so `docker-compose.override.yml` (dev
# bind-mounts + --reload, auto-merged by a bare `docker compose up`) is
# never picked up here by accident. For local development with hot reload,
# run `docker compose up -d` directly instead (no `-f` flag) so the override
# file merges in.
#
# Usage: ./deployment/scripts/deploy_docker_compose.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

COMPOSE="docker compose -f docker-compose.yml"

if [[ ! -f .env ]]; then
  echo "No .env found — copying .env.example. Edit it (especially JWT_SECRET_KEY) before running in production." >&2
  cp .env.example .env
fi

echo "Building images..."
$COMPOSE build

echo "Starting stack (postgres, redis, mlflow, api, frontend)..."
$COMPOSE up -d

echo "Waiting for the API to report healthy..."
for _ in $(seq 1 30); do
  if curl -sf http://localhost:8000/health >/dev/null; then
    echo "API is up: http://localhost:8000"
    echo "Frontend:  http://localhost:8501"
    echo "MLflow UI: http://localhost:5000"
    echo ""
    echo "Optional extras (not started by default):"
    echo "  Admin DB UI:  $COMPOSE --profile tools up -d adminer   (http://localhost:8080)"
    echo "  Training job: $COMPOSE --profile ml run --rm ml-worker"
    exit 0
  fi
  sleep 2
done

echo "API did not become healthy in time — check logs with: $COMPOSE logs api" >&2
exit 1
