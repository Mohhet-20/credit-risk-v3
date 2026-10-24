"""
Database engine/session management.

Design decision: `check_same_thread=False` is only applied for SQLite (the
zero-setup local default); it's a no-op / unnecessary for Postgres, so we
conditionally set connect_args based on the URL scheme rather than always
passing SQLite-specific options that Postgres would reject.

Design decision: `build_engine()` auto-creates the parent directory of a
file-based SQLite URL before `create_engine()` is called. SQLite raises a
bare `unable to open database file` (no mention of the missing directory)
if the parent doesn't exist -- an easy footgun on a fresh checkout or
deployment where `data/processed/` hasn't been created yet. This protects
every entrypoint (API, scripts, tests) that imports this module.

Reliability (this pass): `init_db_with_retry()` retries `init_db()` with
exponential backoff. This matters specifically under `docker compose up`:
the `api` service can start before the `postgres` service has finished
accepting connections even with a compose `depends_on: condition:
service_healthy` (there's still a window right as Postgres flips healthy),
and previously a cold `docker compose up` would sometimes crash-loop the
API container on the very first connection attempt. `api/app.py`'s startup
lifespan calls `init_db_with_retry()` instead of the bare `init_db()`.

Migrations (this pass): `init_db()` now runs `alembic upgrade head`
in-process instead of `Base.metadata.create_all()`. No deployment path
(docker-compose, Kubernetes, the deploy scripts) invoked Alembic at all --
`create_all()` created the right tables on a fresh database but never
stamped `alembic_version`, so the first real schema change shipped as a new
migration would have failed with "table already exists" against every
database that had only ever gone through `create_all()`. Running the
migration chain at startup (env.py already reads the same `DATABASE_URL`
this module does -- see `database/migrations/env.py`) makes every
deployment path go through the one real mechanism instead of a shortcut
that only happened to look equivalent on a fresh database.

That transition itself needs handling: any database that already has these
tables from a prior `create_all()`-based deployment has no `alembic_version`
row, so a bare `alembic upgrade head` against it fails with "table already
exists" (it tries to CREATE TABLE for tables Alembic doesn't know are
already there). `init_db()` detects that specific case -- tables exist, no
`alembic_version` -- and stamps the database at `head` instead of replaying
the CREATE TABLE migrations, the standard way to adopt Alembic onto an
existing unversioned database. A genuinely empty database still goes
through the normal `upgrade head` path and gets both the tables and the
stamp.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from configs.settings import PROJECT_ROOT, get_settings
from core.logging import get_logger

logger = get_logger(__name__)


def _ensure_sqlite_parent_dir(url: str) -> None:
    """Create the parent directory of a file-based SQLite URL if missing.

    In-memory URLs (`sqlite:///:memory:` or `sqlite://`) are left alone.
    """
    if not url.startswith("sqlite:///") or ":memory:" in url:
        return
    db_path = Path(url.removeprefix("sqlite:///"))
    db_path.parent.mkdir(parents=True, exist_ok=True)


def build_engine(database_url: str | None = None):
    settings = get_settings()
    url = database_url or settings.database_url
    _ensure_sqlite_parent_dir(url)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


engine = build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _alembic_config() -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "database" / "migrations"))
    return cfg


def _current_alembic_revision() -> str | None:
    """Returns the current revision recorded in `alembic_version`, or `None`
    if the table doesn't exist OR exists but has no row -- both mean "Alembic
    doesn't know what state this database is in", which is the condition
    that actually matters (not merely whether the table exists: a migration
    run that crashed partway through can leave an empty `alembic_version`
    table behind, since Alembic creates that table before running the first
    migration but only inserts the revision row after it completes)."""
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def init_db() -> None:
    """Applies every pending Alembic migration (`alembic upgrade head`).
    `database/migrations/env.py` reads `DATABASE_URL` from the same
    `configs.settings.get_settings()` this module does, so this migrates
    whatever database the running process is actually configured against --
    SQLite in local dev/tests, Postgres in Docker Compose/Kubernetes/prod.

    If the database already has real tables but Alembic has no recorded
    revision for it (a database that only ever went through the old
    `create_all()` path, or a migration run that crashed before stamping),
    it's stamped at `head` instead of upgraded -- see the module docstring."""
    cfg = _alembic_config()
    existing_tables = set(inspect(engine).get_table_names()) - {"alembic_version"}
    if existing_tables and _current_alembic_revision() is None:
        logger.warning(
            "Database has tables but no recorded Alembic revision -- stamping "
            "at head instead of replaying migrations (adopting Alembic onto "
            "an existing create_all()-created database, or recovering from a "
            "migration run that crashed before stamping).",
            extra={"context": {"existing_tables": sorted(existing_tables)}},
        )
        command.stamp(cfg, "head")
        return
    command.upgrade(cfg, "head")


def init_db_with_retry(max_attempts: int = 5, base_delay_seconds: float = 0.5) -> None:
    """Calls `init_db()`, retrying on `OperationalError` (connection
    refused, DNS not yet resolvable, "database system is starting up", etc.)
    with exponential backoff (0.5s, 1s, 2s, 4s, ...). Raises the last
    exception if every attempt fails, so the process still exits non-zero
    and the orchestrator's restart policy takes over -- this smooths over a
    brief startup race, it does not paper over a genuinely dead database."""
    last_error: OperationalError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            init_db()
            if attempt > 1:
                logger.info(f"Database became reachable on attempt {attempt}/{max_attempts}")
            return
        except OperationalError as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            delay = base_delay_seconds * (2 ** (attempt - 1))
            logger.warning(
                f"Database not reachable yet (attempt {attempt}/{max_attempts}), retrying in {delay:.1f}s",
                extra={"context": {"error": str(exc)}},
            )
            time.sleep(delay)
    logger.error(f"Database unreachable after {max_attempts} attempts, giving up")
    raise last_error


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a request-scoped session, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for use OUTSIDE FastAPI request handlers (scripts, jobs)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
