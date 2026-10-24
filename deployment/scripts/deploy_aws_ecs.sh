#!/usr/bin/env bash
# Reference deployment path for AWS: build -> push to ECR -> deploy to ECS
# Fargate. This is a scripted reference, not a one-command deploy -- ECS
# task/service definitions and the surrounding VPC/ALB/RDS setup are
# expected to already exist (e.g. provisioned via Terraform/CDK) and are
# referenced here by name via environment variables.
#
# Required env vars:
#   AWS_REGION            e.g. us-east-1
#   AWS_ACCOUNT_ID
#   ECR_REPO_API           e.g. credit-risk-api
#   ECR_REPO_FRONTEND      e.g. credit-risk-frontend
#   ECS_CLUSTER
#   ECS_SERVICE_API
#   ECS_SERVICE_FRONTEND
#
# Usage: AWS_REGION=us-east-1 AWS_ACCOUNT_ID=123456789012 ... \
#        ./deployment/scripts/deploy_aws_ecs.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

: "${AWS_REGION:?Set AWS_REGION}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
: "${ECR_REPO_API:?Set ECR_REPO_API}"
: "${ECR_REPO_FRONTEND:?Set ECR_REPO_FRONTEND}"
: "${ECS_CLUSTER:?Set ECS_CLUSTER}"
: "${ECS_SERVICE_API:?Set ECS_SERVICE_API}"
: "${ECS_SERVICE_FRONTEND:?Set ECS_SERVICE_FRONTEND}"

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%s)"

echo "Authenticating to ECR (${ECR_REGISTRY})..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "Building and pushing API image..."
docker build -f Dockerfile.api -t "${ECR_REGISTRY}/${ECR_REPO_API}:${TAG}" .
docker push "${ECR_REGISTRY}/${ECR_REPO_API}:${TAG}"

echo "Building and pushing frontend image..."
docker build -f Dockerfile.frontend -t "${ECR_REGISTRY}/${ECR_REPO_FRONTEND}:${TAG}" .
docker push "${ECR_REGISTRY}/${ECR_REPO_FRONTEND}:${TAG}"

echo "Forcing new ECS deployments..."
aws ecs update-service --cluster "${ECS_CLUSTER}" --service "${ECS_SERVICE_API}" \
  --force-new-deployment --region "${AWS_REGION}" >/dev/null
aws ecs update-service --cluster "${ECS_CLUSTER}" --service "${ECS_SERVICE_FRONTEND}" \
  --force-new-deployment --region "${AWS_REGION}" >/dev/null

echo "Deployed tag ${TAG} to ECS cluster ${ECS_CLUSTER}."
echo "Note: this assumes the ECS task definitions already reference"
echo "${ECR_REGISTRY}/${ECR_REPO_API}:latest / :${ECR_REPO_FRONTEND}:latest (or use a CD"
echo "pipeline step to register a new task definition revision pinned to :${TAG})."
