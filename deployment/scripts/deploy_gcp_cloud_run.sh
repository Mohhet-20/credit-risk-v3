#!/usr/bin/env bash
# Reference deployment path for GCP: build -> push to Artifact Registry ->
# deploy both services to Cloud Run. Cloud Run is a good fit here because
# both the API and the Streamlit frontend are stateless HTTP services --
# state lives in Cloud SQL (Postgres) and a separately hosted MLflow
# instance (Cloud Run also works for MLflow, or use a small Compute Engine
# VM / GKE Autopilot pod if you need a persistent SQLite backend store).
#
# Required env vars:
#   GCP_PROJECT
#   GCP_REGION              e.g. us-central1
#   AR_REPO                 Artifact Registry repo name, e.g. credit-risk
#   DATABASE_URL             postgresql://... (Cloud SQL connection string)
#   JWT_SECRET_KEY
#
# Usage: GCP_PROJECT=my-project GCP_REGION=us-central1 AR_REPO=credit-risk \
#        DATABASE_URL=... JWT_SECRET_KEY=... \
#        ./deployment/scripts/deploy_gcp_cloud_run.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

: "${GCP_PROJECT:?Set GCP_PROJECT}"
: "${GCP_REGION:?Set GCP_REGION}"
: "${AR_REPO:?Set AR_REPO}"
: "${DATABASE_URL:?Set DATABASE_URL}"
: "${JWT_SECRET_KEY:?Set JWT_SECRET_KEY}"

REGISTRY="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO}"
TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%s)"

gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

echo "Building and pushing API image..."
docker build -f Dockerfile.api -t "${REGISTRY}/api:${TAG}" .
docker push "${REGISTRY}/api:${TAG}"

echo "Deploying API to Cloud Run..."
gcloud run deploy credit-risk-api \
  --image "${REGISTRY}/api:${TAG}" \
  --project "${GCP_PROJECT}" --region "${GCP_REGION}" \
  --platform managed --allow-unauthenticated \
  --port 8000 \
  --set-env-vars "APP_ENV=production,DATABASE_URL=${DATABASE_URL},JWT_SECRET_KEY=${JWT_SECRET_KEY},MLFLOW_ALLOW_FILE_STORE=true"

API_URL="$(gcloud run services describe credit-risk-api \
  --project "${GCP_PROJECT}" --region "${GCP_REGION}" --format='value(status.url)')"
echo "API deployed at ${API_URL}"

echo "Building and pushing frontend image..."
docker build -f Dockerfile.frontend -t "${REGISTRY}/frontend:${TAG}" .
docker push "${REGISTRY}/frontend:${TAG}"

echo "Deploying frontend to Cloud Run..."
gcloud run deploy credit-risk-frontend \
  --image "${REGISTRY}/frontend:${TAG}" \
  --project "${GCP_PROJECT}" --region "${GCP_REGION}" \
  --platform managed --allow-unauthenticated \
  --port 8501 \
  --set-env-vars "API_BASE_URL=${API_URL}"

echo "Deployed. Frontend URL:"
gcloud run services describe credit-risk-frontend \
  --project "${GCP_PROJECT}" --region "${GCP_REGION}" --format='value(status.url)'
