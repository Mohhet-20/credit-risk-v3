#!/usr/bin/env bash
# Reference deployment path for Azure: build -> push to Azure Container
# Registry -> deploy both services to Azure Container Apps.
#
# Required env vars:
#   AZ_RESOURCE_GROUP
#   AZ_ACR_NAME              Container Registry name (no domain suffix)
#   AZ_ENV_NAME               Container Apps environment name
#   AZ_LOCATION               e.g. eastus
#   DATABASE_URL               postgresql://... (Azure Database for PostgreSQL)
#   JWT_SECRET_KEY
#
# Usage: AZ_RESOURCE_GROUP=credit-risk-rg AZ_ACR_NAME=creditriskacr \
#        AZ_ENV_NAME=credit-risk-env AZ_LOCATION=eastus \
#        DATABASE_URL=... JWT_SECRET_KEY=... \
#        ./deployment/scripts/deploy_azure_container_apps.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

: "${AZ_RESOURCE_GROUP:?Set AZ_RESOURCE_GROUP}"
: "${AZ_ACR_NAME:?Set AZ_ACR_NAME}"
: "${AZ_ENV_NAME:?Set AZ_ENV_NAME}"
: "${AZ_LOCATION:?Set AZ_LOCATION}"
: "${DATABASE_URL:?Set DATABASE_URL}"
: "${JWT_SECRET_KEY:?Set JWT_SECRET_KEY}"

ACR_LOGIN_SERVER="${AZ_ACR_NAME}.azurecr.io"
TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%s)"

echo "Logging in to ACR (${ACR_LOGIN_SERVER})..."
az acr login --name "${AZ_ACR_NAME}"

echo "Building and pushing API image..."
docker build -f Dockerfile.api -t "${ACR_LOGIN_SERVER}/credit-risk-api:${TAG}" .
docker push "${ACR_LOGIN_SERVER}/credit-risk-api:${TAG}"

echo "Building and pushing frontend image..."
docker build -f Dockerfile.frontend -t "${ACR_LOGIN_SERVER}/credit-risk-frontend:${TAG}" .
docker push "${ACR_LOGIN_SERVER}/credit-risk-frontend:${TAG}"

echo "Deploying/updating API container app..."
az containerapp up \
  --name credit-risk-api \
  --resource-group "${AZ_RESOURCE_GROUP}" \
  --environment "${AZ_ENV_NAME}" \
  --location "${AZ_LOCATION}" \
  --image "${ACR_LOGIN_SERVER}/credit-risk-api:${TAG}" \
  --target-port 8000 --ingress external \
  --env-vars "APP_ENV=production" "DATABASE_URL=${DATABASE_URL}" \
             "JWT_SECRET_KEY=${JWT_SECRET_KEY}" "MLFLOW_ALLOW_FILE_STORE=true"

API_FQDN="$(az containerapp show --name credit-risk-api --resource-group "${AZ_RESOURCE_GROUP}" \
  --query properties.configuration.ingress.fqdn -o tsv)"
echo "API deployed at https://${API_FQDN}"

echo "Deploying/updating frontend container app..."
az containerapp up \
  --name credit-risk-frontend \
  --resource-group "${AZ_RESOURCE_GROUP}" \
  --environment "${AZ_ENV_NAME}" \
  --location "${AZ_LOCATION}" \
  --image "${ACR_LOGIN_SERVER}/credit-risk-frontend:${TAG}" \
  --target-port 8501 --ingress external \
  --env-vars "API_BASE_URL=https://${API_FQDN}"

echo "Deployed."
az containerapp show --name credit-risk-frontend --resource-group "${AZ_RESOURCE_GROUP}" \
  --query properties.configuration.ingress.fqdn -o tsv
