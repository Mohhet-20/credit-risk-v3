"""
Centralized application settings using pydantic-settings.

Design decision: a single `Settings` object sourced from environment
variables (with a `.env` fallback for local dev) is the standard pattern for
12-factor apps. It means secrets never live in code, the same image can be
promoted dev -> staging -> prod by swapping env vars only, and every module
gets IDE-autocompleted, type-checked config instead of scattered `os.getenv`
calls with silent typos.

Secrets validation (this pass): `Settings` no longer silently accepts an
insecure JWT secret in a production deployment. `INSECURE_DEFAULT_JWT_SECRET`
is the exact fallback value used throughout local dev/tests; but the repo
also ships *other* placeholder values in `.env.example`, `docker-compose.yml`,
and `deployment/kubernetes/02-secret.yaml`, several of which are long enough
to pass a naive length check while still being publicly-committed and
guessable. `Settings.validate_production_secrets()` therefore also rejects
any secret containing a common placeholder marker (e.g. "change-me",
"replace_me", "insecure", "example") regardless of length, in addition to
the exact-default and minimum-length checks. This is a fail-fast check
called explicitly from `api/app.py`'s startup lifespan (not a pydantic
`field_validator` on the field itself) so that importing `Settings` in a
test or dev context never raises just because `APP_ENV` isn't set to
`production` — the check is specifically "is this safe to run in prod",
not "is this value well-formed."
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INSECURE_DEFAULT_JWT_SECRET = "dev-only-insecure-secret"
MIN_PRODUCTION_SECRET_LENGTH = 32

# Substrings that mark a secret as an unedited placeholder regardless of its
# length. Covers every placeholder value shipped in this repo (.env.example,
# docker-compose.yml, deployment/kubernetes/02-secret.yaml) plus common
# conventions other placeholders tend to follow.
INSECURE_SECRET_MARKERS = (
    "change-me",
    "changeme",
    "replace_me",
    "replace-me",
    "insecure",
    "example",
    "placeholder",
    "dev-only",
    "your-secret",
    "todo",
)


class InsecureConfigurationError(RuntimeError):
    """Raised when a production deployment is about to start with a
    configuration value that is unsafe outside local development (e.g. the
    shipped default JWT signing secret). Deliberately a plain `RuntimeError`
    subclass, not a `CreditRiskPlatformError` -- this is a startup/deploy
    concern the API's request-level exception handlers never see."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Credit Risk Intelligence Platform"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # --- Security ---
    jwt_secret_key: str = Field(default=INSECURE_DEFAULT_JWT_SECRET)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # --- Database ---
    database_url: str = Field(default=f"sqlite:///{PROJECT_ROOT / 'data' / 'processed' / 'app.db'}")

    # --- MLflow ---
    mlflow_tracking_uri: str = str(PROJECT_ROOT / "mlruns")
    mlflow_experiment_name: str = "credit-risk-v2"
    mlflow_registry_model_name: str = "credit_risk_champion"

    # --- API ---
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:3000"]
    rate_limit_per_minute: int = 60

    # --- Cache / Redis ---
    # `redis_url` is intentionally optional (default: unset). When unset,
    # `core/cache.py::get_cache()` uses the in-process `InMemoryTTLCache` --
    # correct for local dev, tests, and a single-replica deployment. Setting
    # `REDIS_URL` (wired up as the `redis` service in docker-compose.yml, or
    # a managed Redis/ElastiCache/Memorystore endpoint in cloud deployments)
    # is what triggers the swap to a shared, multi-replica-safe cache -- see
    # the design-decision docstring at the top of `core/cache.py`.
    redis_url: str | None = None
    cache_key_prefix: str = "credit-risk:"

    # --- Model / paths ---
    model_artifact_dir: Path = PROJECT_ROOT / "models" / "artifacts"
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    reports_dir: Path = PROJECT_ROOT / "reports"
    decision_threshold: float = 0.5

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    def validate_production_secrets(self) -> list[str]:
        """Returns a list of human-readable problems with the current
        configuration that make it unsafe to run as `app_env=production`.
        An empty list means the configuration passed. Split out from a
        `raise`-on-failure method so callers can choose to log-and-continue
        (e.g. a diagnostics endpoint) as well as fail fast (the startup
        lifespan in `api/app.py`, via `validate_or_raise()`)."""
        if not self.is_production:
            return []
        problems = []
        secret_lower = self.jwt_secret_key.lower()
        if self.jwt_secret_key == INSECURE_DEFAULT_JWT_SECRET:
            problems.append(
                "JWT_SECRET_KEY is still set to the shipped development default. "
                "Generate a unique secret (e.g. `openssl rand -hex 32`) and set it "
                "via the JWT_SECRET_KEY environment variable before running in production."
            )
        elif any(marker in secret_lower for marker in INSECURE_SECRET_MARKERS):
            problems.append(
                "JWT_SECRET_KEY looks like an unedited placeholder value (matches a "
                "known placeholder pattern such as 'change-me' or 'example'). Generate "
                "a unique secret (e.g. `openssl rand -hex 32`) before running in production."
            )
        elif len(self.jwt_secret_key) < MIN_PRODUCTION_SECRET_LENGTH:
            problems.append(
                f"JWT_SECRET_KEY is only {len(self.jwt_secret_key)} characters; "
                f"production requires at least {MIN_PRODUCTION_SECRET_LENGTH} for "
                "adequate HMAC signing strength."
            )
        if self.debug:
            problems.append(
                "DEBUG is true in a production environment -- this suppresses the "
                "Strict-Transport-Security header (see api/middleware.py) and may "
                "leak stack traces via framework default error pages."
            )
        if self.database_url.startswith("sqlite"):
            problems.append(
                "DATABASE_URL points at SQLite in a production environment -- SQLite "
                "does not support concurrent writers safely across multiple API "
                "replicas. Use PostgreSQL (see docker-compose.yml's `postgres` service)."
            )
        problems.extend(self._database_password_problems())
        return problems

    def _database_password_problems(self) -> list[str]:
        """Checks the password embedded in a `postgresql://user:pass@host/db`
        DATABASE_URL against the same placeholder/length rules as the JWT
        secret. Unlike `jwt_secret_key`, there's no dedicated settings field
        for this -- `docker-compose.yml`/`deployment/kubernetes/02-secret.yaml`
        build DATABASE_URL from `POSTGRES_PASSWORD`, so the weak shipped
        default (`credit_risk`/`change-me`) reaches this app only as part of
        the connection string, where nothing previously inspected it."""
        parsed = urlparse(self.database_url)
        if parsed.scheme not in ("postgresql", "postgres") or not parsed.password:
            return []
        password_lower = parsed.password.lower()
        if any(marker in password_lower for marker in INSECURE_SECRET_MARKERS):
            return [
                "DATABASE_URL's password looks like an unedited placeholder value "
                "(matches a known placeholder pattern such as 'change-me'). Set a "
                "unique POSTGRES_PASSWORD before running in production."
            ]
        if len(parsed.password) < MIN_PRODUCTION_SECRET_LENGTH or parsed.password == parsed.username:
            return [
                "DATABASE_URL's password is too weak for production (too short, or "
                "identical to the username). Set a unique, sufficiently long "
                "POSTGRES_PASSWORD before running in production."
            ]
        return []

    def validate_or_raise(self) -> None:
        """Fail-fast entry point: raises `InsecureConfigurationError` if
        `validate_production_secrets()` found any problems. Called once from
        `api/app.py`'s startup lifespan so a misconfigured production
        deployment refuses to start (and serve traffic insecurely) rather
        than logging a warning that operators may not see."""
        problems = self.validate_production_secrets()
        if problems:
            joined = " | ".join(problems)
            raise InsecureConfigurationError(f"Refusing to start with an insecure production configuration: {joined}")


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor — avoids re-parsing env vars on every call."""
    return Settings()
