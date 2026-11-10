# syntax=docker/dockerfile:1
#
# ML services image: training pipeline runs and other batch/background
# jobs (retraining, scheduled evaluation, data imports), kept as a SEPARATE
# image from `Dockerfile.api` even though they currently share
# `requirements.txt` and most of the codebase.
#
# Design decision: separating "serves live HTTP traffic" (Dockerfile.api)
# from "runs a batch job to completion and exits" (this image) matters
# operationally even before the dependency sets diverge -- they have
# different resource profiles (training wants more CPU/memory and no
# uptime SLA; the API wants low, predictable latency and high availability),
# different restart policies (`restart: unless-stopped` for the API,
# `restart: "no"` / run-to-completion for a training job), and different
# health semantics (an HTTP healthcheck makes no sense for a process with no
# server socket). Keeping them as one image would force every training run
# to carry uvicorn/fastapi/slowapi's runtime footprint for no reason, and
# vice versa.
#
# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: runtime ----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_ALLOW_FILE_STORE=true \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

COPY . .

RUN mkdir -p data/raw data/processed/checkpoints models/artifacts mlruns reports \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app /opt/venv
USER appuser

# No HEALTHCHECK: this image runs a batch job to completion (training,
# retraining, a scheduled evaluation/cleanup script) rather than serving an
# HTTP endpoint, so there's no port to probe -- container exit code IS the
# health signal (`docker inspect --format='{{.State.ExitCode}}'` / a
# non-zero exit fails the orchestrator's job, exactly like a k8s `Job`
# rather than a `Deployment`).
#
# Default command runs the full training pipeline; override `command:` in
# docker-compose (or a k8s Job's `args:`) to run a different stage
# (`--stage tune --model xgboost`) or a different batch script entirely
# (e.g. `python3 -m pipelines.run_training --stage finalize`).
CMD ["python3", "-m", "pipelines.run_training", "--stage", "all"]
