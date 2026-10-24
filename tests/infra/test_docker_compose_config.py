"""
Structural validation of the Docker Compose files -- runnable without a
Docker daemon or the `docker compose` CLI (neither is available in this
sandbox, and a CI runner shouldn't need one just to catch a typo'd service
name or a missing healthcheck). These tests parse the YAML directly and
assert on the shape, the same properties `docker compose config` would
otherwise catch by actually resolving the file.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load(filename: str) -> dict:
    with open(PROJECT_ROOT / filename) as f:
        return yaml.safe_load(f)


def test_base_compose_file_is_valid_yaml_with_expected_top_level_keys():
    compose = _load("docker-compose.yml")
    assert set(compose.keys()) >= {"services", "networks", "volumes"}


def test_base_compose_defines_all_expected_services():
    compose = _load("docker-compose.yml")
    expected = {"postgres", "redis", "mlflow", "api", "frontend", "web", "ml-worker", "adminer"}
    assert expected <= set(compose["services"].keys())


def test_base_compose_defines_isolated_frontend_and_backend_networks():
    compose = _load("docker-compose.yml")
    assert set(compose["networks"].keys()) == {"backend", "frontend"}

    # Postgres and Redis must NOT be reachable from the `frontend` network --
    # that's the entire point of the network split (see the module docstring
    # in docker-compose.yml).
    for service_name in ("postgres", "redis", "mlflow"):
        service = compose["services"][service_name]
        assert service.get("networks") == ["backend"], f"{service_name} must be on the backend network only"

    # The API is the sole bridge between the two networks.
    api_networks = set(compose["services"]["api"].get("networks", []))
    assert api_networks == {"backend", "frontend"}

    # The frontend must not have a direct path to the backend network.
    frontend_networks = set(compose["services"]["frontend"].get("networks", []))
    assert frontend_networks == {"frontend"}


def test_base_compose_declares_expected_named_volumes():
    compose = _load("docker-compose.yml")
    assert set(compose["volumes"].keys()) == {"postgres_data", "redis_data", "mlflow_data", "model_artifacts"}


def test_every_long_running_service_has_a_healthcheck():
    compose = _load("docker-compose.yml")
    # ml-worker and adminer are excluded: ml-worker is a run-to-completion
    # batch job (see Dockerfile.ml's design-decision comment -- container
    # exit code is the health signal, not an HTTP probe), and adminer is a
    # third-party image whose upstream Dockerfile doesn't ship one.
    long_running = {"postgres", "redis", "mlflow", "api", "frontend", "web"}
    for name in long_running:
        assert "healthcheck" in compose["services"][name], f"{name} is missing a healthcheck"


def test_optional_services_are_gated_behind_profiles_not_started_by_default():
    compose = _load("docker-compose.yml")
    assert compose["services"]["ml-worker"]["profiles"] == ["ml"]
    assert compose["services"]["adminer"]["profiles"] == ["tools"]
    # The five core services must NOT carry a `profiles` key -- that's what
    # makes them start on a bare `docker compose up` with no --profile flag.
    for name in ("postgres", "redis", "mlflow", "api", "frontend"):
        assert "profiles" not in compose["services"][name]


def test_api_depends_on_its_datastores_with_explicit_health_conditions():
    compose = _load("docker-compose.yml")
    api_depends_on = compose["services"]["api"]["depends_on"]
    assert api_depends_on["postgres"]["condition"] == "service_healthy"
    assert api_depends_on["redis"]["condition"] == "service_healthy"


def test_api_receives_redis_url_pointing_at_the_compose_service():
    compose = _load("docker-compose.yml")
    assert compose["services"]["api"]["environment"]["REDIS_URL"] == "redis://redis:6379/0"


def test_every_service_declares_resource_limits():
    compose = _load("docker-compose.yml")
    for name, service in compose["services"].items():
        limits = service.get("deploy", {}).get("resources", {}).get("limits")
        assert limits, f"{name} is missing deploy.resources.limits"
        assert "memory" in limits and "cpus" in limits


def test_dev_override_file_is_valid_yaml_and_only_touches_api_and_frontend():
    override = _load("docker-compose.override.yml")
    assert set(override["services"].keys()) == {"api", "frontend"}


def test_dev_override_enables_hot_reload_and_bind_mounts_source():
    override = _load("docker-compose.override.yml")
    api_override = override["services"]["api"]
    assert "--reload" in api_override["command"]
    assert any(v.startswith("./:/app") for v in api_override["volumes"])


def test_test_compose_file_defines_a_gated_tests_service():
    test_compose = _load("docker-compose.test.yml")
    tests_service = test_compose["services"]["tests"]
    assert tests_service["profiles"] == ["test"]
    assert tests_service["build"]["dockerfile"] == "Dockerfile.api"
    assert "pytest" in tests_service["entrypoint"]
