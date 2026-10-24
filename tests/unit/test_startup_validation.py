"""
Tests for `configs/settings.py`'s production secrets validation
(`Settings.validate_production_secrets()` / `validate_or_raise()`) and
`database/session.py::init_db_with_retry()`'s retry/backoff behavior --
both new reliability/security surfaces introduced to make `api/app.py`'s
startup lifespan fail fast on misconfiguration and resilient to a
database that isn't ready yet (e.g. a `docker compose up` cold start).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from configs.settings import INSECURE_DEFAULT_JWT_SECRET, InsecureConfigurationError, Settings
from database.session import init_db_with_retry


def _settings(**overrides) -> Settings:
    defaults = dict(
        app_env="development",
        jwt_secret_key="a" * 40,
        debug=False,
        database_url=f"postgresql://produser:{'p' * 40}@host/db",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_non_production_settings_skip_validation_entirely():
    s = _settings(app_env="development", jwt_secret_key=INSECURE_DEFAULT_JWT_SECRET, debug=True)
    assert s.validate_production_secrets() == []
    s.validate_or_raise()  # must not raise


def test_production_with_default_secret_is_rejected():
    s = _settings(app_env="production", jwt_secret_key=INSECURE_DEFAULT_JWT_SECRET)
    problems = s.validate_production_secrets()
    assert any("default" in p.lower() for p in problems)
    with pytest.raises(InsecureConfigurationError):
        s.validate_or_raise()


def test_production_with_short_secret_is_rejected():
    s = _settings(app_env="production", jwt_secret_key="short")
    problems = s.validate_production_secrets()
    assert any("characters" in p for p in problems)


@pytest.mark.parametrize(
    "placeholder",
    [
        "change-me-in-production-use-a-real-secret",  # .env.example
        "change-me-in-production",  # docker-compose.yml default
        "REPLACE_ME_WITH_A_STRONG_RANDOM_SECRET",  # deployment/kubernetes/02-secret.yaml
    ],
)
def test_production_with_shipped_placeholder_secret_is_rejected(placeholder):
    """These are the exact placeholder strings shipped in this repo's own
    .env.example / docker-compose.yml / k8s secret manifest. Some are long
    enough to pass MIN_PRODUCTION_SECRET_LENGTH on their own, so the length
    check alone isn't sufficient -- the placeholder-marker check must catch
    them too."""
    s = _settings(app_env="production", jwt_secret_key=placeholder)
    problems = s.validate_production_secrets()
    assert any("placeholder" in p.lower() for p in problems)
    with pytest.raises(InsecureConfigurationError):
        s.validate_or_raise()


def test_production_with_debug_true_is_rejected():
    s = _settings(app_env="production", debug=True)
    problems = s.validate_production_secrets()
    assert any("DEBUG" in p for p in problems)


def test_production_with_sqlite_is_rejected():
    s = _settings(app_env="production", database_url="sqlite:///data/processed/app.db")
    problems = s.validate_production_secrets()
    assert any("SQLite" in p for p in problems)


def test_production_with_fully_safe_config_passes():
    s = _settings(app_env="production")
    assert s.validate_production_secrets() == []
    s.validate_or_raise()  # must not raise


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://credit_risk:credit_risk@postgres:5432/credit_risk",  # docker-compose.yml default
        "postgresql://produser:change-me@postgres:5432/db",  # placeholder marker
        "postgresql://produser:short@postgres:5432/db",  # too short
    ],
)
def test_production_with_weak_postgres_password_is_rejected(database_url):
    """DATABASE_URL's password is the shipped docker-compose.yml default
    (identical to the username) or another weak/placeholder value -- this
    reaches the app only embedded in the connection string (there's no
    separate settings field for it), so nothing previously inspected it."""
    s = _settings(app_env="production", database_url=database_url)
    problems = s.validate_production_secrets()
    assert any("password" in p.lower() for p in problems)
    with pytest.raises(InsecureConfigurationError):
        s.validate_or_raise()


def test_production_with_strong_postgres_password_passes():
    s = _settings(app_env="production", database_url=f"postgresql://produser:{'x' * 40}@host/db")
    assert s.validate_production_secrets() == []


def test_production_reports_all_problems_at_once():
    """Fail-fast should surface every problem in one pass, not just the
    first -- an operator fixing one env var at a time via trial-and-error
    against a restart loop is a worse experience than seeing the full list."""
    s = _settings(
        app_env="production",
        jwt_secret_key=INSECURE_DEFAULT_JWT_SECRET,
        debug=True,
        database_url="sqlite:///x.db",
    )
    problems = s.validate_production_secrets()
    assert len(problems) == 3


def test_init_db_with_retry_succeeds_immediately_when_db_is_healthy():
    with patch("database.session.init_db") as mock_init:
        init_db_with_retry(max_attempts=3, base_delay_seconds=0.01)
    mock_init.assert_called_once()


def test_init_db_with_retry_recovers_after_transient_failures():
    call_count = 0

    def flaky_init():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise OperationalError("connect failed", {}, Exception("refused"))

    with patch("database.session.init_db", side_effect=flaky_init):
        with patch("database.session.time.sleep") as mock_sleep:  # no real delay in tests
            init_db_with_retry(max_attempts=5, base_delay_seconds=0.01)

    assert call_count == 3
    assert mock_sleep.call_count == 2


def test_init_db_with_retry_raises_after_exhausting_all_attempts():
    always_fails = MagicMock(side_effect=OperationalError("connect failed", {}, Exception("refused")))
    with patch("database.session.init_db", always_fails):
        with patch("database.session.time.sleep"):
            with pytest.raises(OperationalError):
                init_db_with_retry(max_attempts=3, base_delay_seconds=0.01)
    assert always_fails.call_count == 3
