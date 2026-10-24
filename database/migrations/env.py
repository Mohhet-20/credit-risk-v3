"""
Alembic environment script.

Design decision: `target_metadata` points at `database.models.Base.metadata`
(this project's single source of truth for schema) and the connection URL is
pulled from `configs.settings.get_settings()` at runtime rather than
hard-coded in `alembic.ini` -- so the exact same migration set applies
against SQLite (local dev) or PostgreSQL (staging/production) without
editing `alembic.ini` per environment; it's driven by the same
`DATABASE_URL` environment variable every other part of the platform
already reads, including `database/session.py::init_db()`, which runs
`alembic upgrade head` through this same env.py on every API startup.
`alembic.ini`'s own `sqlalchemy.url` line is left as a placeholder and is
intentionally never used.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the project root importable (this file lives at database/migrations/env.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.settings import get_settings  # noqa: E402
from database.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection (`alembic upgrade
    head --sql`) -- useful for review before applying against a production
    database, or for environments where the migration runner shouldn't hold
    live DB credentials."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
