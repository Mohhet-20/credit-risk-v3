#!/usr/bin/env bash
# Applies the deployment/kubernetes manifests, in order, to whatever cluster
# your current kubectl context points at.
#
# Prerequisites:
#   - kubectl configured against the target cluster
#   - deployment/kubernetes/02-secret.yaml edited with real values (or
#     replaced with an External Secrets Operator resource) BEFORE applying
#   - api/frontend/mlflow images built and pushed to a registry your cluster
#     can pull from, with deployment/kubernetes/05-api.yaml, 06-frontend.yaml,
#     and 04-mlflow.yaml updated to reference those image tags
#
# Usage: ./deployment/scripts/deploy_kubernetes.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MANIFEST_DIR="deployment/kubernetes"

echo "Applying manifests from ${MANIFEST_DIR} ..."
for f in "${MANIFEST_DIR}"/*.yaml; do
  echo "  kubectl apply -f ${f}"
  kubectl apply -f "${f}"
done

echo "Waiting for rollout..."
kubectl -n credit-risk rollout status deployment/postgres --timeout=120s
kubectl -n credit-risk rollout status deployment/redis --timeout=120s
kubectl -n credit-risk rollout status deployment/mlflow --timeout=120s
kubectl -n credit-risk rollout status deployment/api --timeout=180s
kubectl -n credit-risk rollout status deployment/frontend --timeout=180s

echo "Done. Check ingress hosts in deployment/kubernetes/07-ingress.yaml for external URLs,"
echo "or run: kubectl -n credit-risk port-forward svc/frontend 8501:8501"
