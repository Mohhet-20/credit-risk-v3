"""
Tests for the Alembic migration setup (`alembic.ini`, `database/migrations/`).

Design decision: this runs Alembic's `command.upgrade()` API directly
against a throwaway temp-file SQLite database (not `:memory:` -- Alembic's
`env.py` opens its own connection, which would see an *empty* database if
we used `:memory:`, since each new connection to an in-memory SQLite URL is
a fresh, unrelated database) rather than shelling out to `alembic` as a
subprocess -- faster, and keeps the test in-process with the rest of the
suite. This is what actually proves the previously-unused `alembic`
dependency (pinned in requirements.txt since an earlier pass but never
wired to an actual migration) now does real, working schema management --
see docs/OPERATIONS.md's "Database migrations" section.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from alembic import command
from alembic.config import Config

from database.models import Base

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _alembic_config(database_url: str, monkeypatch, request) -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "database" / "migrations"))
    # env.py reads DATABASE_URL via configs.settings.get_settings(), which is
    # itself lru_cache'd -- point it at our temp DB via the environment
    # rather than the Config object (env.py never reads sqlalchemy.url).
    # Using monkeypatch (not raw os.environ mutation) so DATABASE_URL is
    # automatically restored at test teardown, plus an explicit finalizer to
    # clear the settings cache again afterward -- otherwise every OTHER test
    # in the same pytest process would keep reading this test's temp
    # DATABASE_URL (and a deleted tempdir) after this test's tempdir is gone.
    from configs.settings import get_settings

    monkeypatch.setenv("DATABASE_URL", database_url)
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)
    return cfg


def test_alembic_upgrade_head_creates_every_model_table(monkeypatch, request):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "migration_test.db"
        database_url = f"sqlite:///{db_path}"
        cfg = _alembic_config(database_url, monkeypatch, request)

        command.upgrade(cfg, "head")

        import sqlite3

        conn = sqlite3.connect(str(db_path))
        try:
            actual_tables = {
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
        finally:
            conn.close()

        expected_tables = set(Base.metadata.tables.keys())
        assert expected_tables.issubset(actual_tables)
        # Alembic's own bookkeeping table proves the migration actually ran
        # through its version-tracking machinery, not just an incidental
        # create_all() side effect.
        assert "alembic_version" in actual_tables


def test_alembic_downgrade_removes_tables(monkeypatch, request):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "migration_downgrade_test.db"
        database_url = f"sqlite:///{db_path}"
        cfg = _alembic_config(database_url, monkeypatch, request)

        command.upgrade(cfg, "head")
        command.downgrade(cfg, "base")

        import sqlite3

        conn = sqlite3.connect(str(db_path))
        try:
            remaining_tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name != 'alembic_version'",
                ).fetchall()
            }
        finally:
            conn.close()

        model_tables = set(Base.metadata.tables.keys())
        assert not (remaining_tables & model_tables)
