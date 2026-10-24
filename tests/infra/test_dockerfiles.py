"""
Structural validation of the Dockerfiles -- runnable without a Docker
daemon (none is available in this sandbox, and shouldn't be required just
to catch "someone deleted the non-root USER line" in CI). These are text-
level assertions on Dockerfile content, not a build; `docker build` itself
is exercised by the `build-images` job in `.github/workflows/ci.yml` where
a real daemon is available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SERVING_DOCKERFILES = ["Dockerfile.api", "Dockerfile.frontend"]
ALL_DOCKERFILES = [*SERVING_DOCKERFILES, "Dockerfile.ml"]


def _read(filename: str) -> str:
    return (PROJECT_ROOT / filename).read_text()


@pytest.mark.parametrize("filename", ALL_DOCKERFILES)
def test_dockerfile_declares_buildkit_syntax(filename):
    content = _read(filename)
    assert content.startswith("# syntax=docker/dockerfile:1")


@pytest.mark.parametrize("filename", ALL_DOCKERFILES)
def test_dockerfile_is_multi_stage_with_named_stages(filename):
    content = _read(filename)
    assert (
        content.count("\nFROM ") + (1 if content.startswith("FROM ") else 0) >= 2
        or len([line for line in content.splitlines() if line.startswith("FROM ")]) >= 2
    )
    assert "AS builder" in content
    assert "AS runtime" in content


@pytest.mark.parametrize("filename", ALL_DOCKERFILES)
def test_dockerfile_runs_as_non_root_user(filename):
    content = _read(filename)
    assert "USER appuser" in content
    assert "useradd" in content


@pytest.mark.parametrize("filename", ALL_DOCKERFILES)
def test_dockerfile_builder_stage_never_reaches_runtime(filename):
    """The venv-copy pattern (`COPY --from=builder /opt/venv /opt/venv`) is
    the entire payoff of the multi-stage split -- assert it's actually
    there, not just that two FROM lines exist."""
    content = _read(filename)
    assert "COPY --from=builder /opt/venv /opt/venv" in content


@pytest.mark.parametrize("filename", SERVING_DOCKERFILES)
def test_serving_dockerfile_has_a_healthcheck(filename):
    content = _read(filename)
    assert "HEALTHCHECK" in content


def test_ml_dockerfile_intentionally_has_no_healthcheck():
    """`Dockerfile.ml` runs a batch job to completion rather than serving
    an HTTP endpoint -- there's no port to probe, and the design decision
    is documented directly in the file (see its own comments). This checks
    for the actual `HEALTHCHECK` *instruction* (a line starting with it),
    not the bare substring -- the file's own explanatory comment mentions
    "No HEALTHCHECK" by name, which would otherwise false-fail a naive
    substring check."""
    content = _read("Dockerfile.ml")
    instruction_lines = [line for line in content.splitlines() if line.startswith("HEALTHCHECK")]
    assert instruction_lines == []
    assert "No HEALTHCHECK" in content  # the design-decision comment is present


def test_api_dockerfile_exposes_port_8000():
    assert "EXPOSE 8000" in _read("Dockerfile.api")


def test_frontend_dockerfile_exposes_port_8501():
    assert "EXPOSE 8501" in _read("Dockerfile.frontend")


@pytest.mark.parametrize("filename", ALL_DOCKERFILES)
def test_dockerfile_sets_python_runtime_env_vars(filename):
    content = _read(filename)
    assert "PYTHONDONTWRITEBYTECODE=1" in content
    assert "PYTHONUNBUFFERED=1" in content


@pytest.mark.parametrize("filename", ALL_DOCKERFILES)
def test_dockerfile_has_a_cmd_instruction(filename):
    content = _read(filename)
    assert "\nCMD [" in content


def test_web_dockerfile_is_multi_stage_non_root_with_healthcheck():
    content = _read("Dockerfile.web")
    assert content.startswith("# syntax=docker/dockerfile:1")
    for stage in ("AS deps", "AS builder", "AS runtime"):
        assert stage in content
    assert "USER app" in content
    assert "HEALTHCHECK" in content
    assert "EXPOSE 3000" in content
    assert ".next/standalone" in content
