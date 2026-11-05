# syntax=docker/dockerfile:1
# Hosted public demo API (render.yaml), sized for a 512 MB free instance. Build context: the repository root.
#
# Serves the exact champion behind the README and reports/MODEL_VALIDATION.md (deploy/demo/champion, produced by
# `make train`), so nothing is retrained on the free host. At build time scripts/demo_build.py scores 300
# out-of-time applications through the API for the dashboard; scripts/demo_start.py copies that showcase fresh on
# every start, on SQLite (an explicit single-instance opt-in), with guest access on and self-registration off.
# Python 3.11 matches the environment that trained the champion (its pickles are version-sensitive).
# For a real deployment use Dockerfile.api with PostgreSQL.

FROM python:3.11-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

FROM python:3.11-slim AS runtime-base
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MLFLOW_ALLOW_FILE_STORE=true \
    OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2
WORKDIR /app
COPY . .

FROM runtime-base AS showcase
# The build drives the API through FastAPI's TestClient, which needs httpx (a dev requirement, pinned in
# requirements-dev.txt). Installed in this stage only; the runtime image does not carry it.
RUN pip install --no-cache-dir httpx==0.28.1
RUN python scripts/demo_build.py && mkdir -p /app/showcase && cp -a /tmp/credit-demo/. /app/showcase/

FROM runtime-base AS runtime
COPY --from=showcase /app/showcase /app/showcase
RUN useradd --create-home --uid 1000 appuser
USER appuser
ENV CREDIT_SHOWCASE_DIR=/app/showcase CREDIT_DEMO_ROOT=/tmp/credit-demo
EXPOSE 8000
CMD ["python", "scripts/demo_start.py"]
