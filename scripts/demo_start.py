"""Start the hosted public demo (deploy/demo/Dockerfile): fresh showcase, locked accounts, then the API.

On every start this:

1. copies the showcase built by scripts/demo_build.py (model registry + scored applications) to a fresh working
   directory, so each restart is a clean demo;
2. runs in production mode on SQLite, which the settings allow only with ALLOW_SQLITE_IN_PRODUCTION: this is a
   single-instance demo whose data is disposable (real deployments use PostgreSQL);
3. gives every existing account (the build's administrator included) a random password nobody knows, turns
   self-registration off and guest access on ("Explore the live demo", a risk analyst);
4. when CREDIT_ADMIN_USERNAME and CREDIT_ADMIN_PASSWORD are set as hosting secrets, creates the owner's administrator;
5. replaces itself with uvicorn on $PORT.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = Path(os.environ.get("CREDIT_SHOWCASE_DIR", "/app/showcase"))
WORKING = Path(os.environ.get("CREDIT_DEMO_ROOT", "/tmp/credit-demo"))


def main() -> int:
    if not (SHOWCASE / "app.db").is_file():
        print(f"[demo] {SHOWCASE} has no app.db; the image build did not produce the showcase", file=sys.stderr)
        return 2
    if not os.environ.get("JWT_SECRET_KEY"):
        print("[demo] JWT_SECRET_KEY is not set; refusing to start a public API without a signing key", file=sys.stderr)
        return 2
    shutil.rmtree(WORKING, ignore_errors=True)  # previous run's database (and SQLite journals) and registry
    shutil.copytree(SHOWCASE, WORKING)
    # A writable reports directory (async batch results land here), seeded with the validation report it serves.
    shutil.copytree(ROOT / "reports", WORKING / "reports", dirs_exist_ok=True)
    os.environ.update(
        {
            "APP_ENV": "production",
            "DEBUG": "false",
            "ALLOW_SQLITE_IN_PRODUCTION": "true",
            "DATABASE_URL": f"sqlite:///{(WORKING / 'app.db').as_posix()}",
            "MODEL_ARTIFACT_DIR": str(WORKING / "artifacts"),
            "REPORTS_DIR": str(WORKING / "reports"),
            "MLFLOW_ALLOW_FILE_STORE": "true",
            "MLFLOW_TRACKING_URI": str(WORKING / "mlruns"),
            "GUEST_ACCESS_ENABLED": "true",
            "REGISTRATION_ENABLED": "false",
        }
    )
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    from sqlalchemy import func

    from core.security import hash_password
    from database.models import Prediction, User
    from database.session import SessionLocal
    from repositories.user_repository import UserRepository

    db = SessionLocal()
    try:
        # Monitoring looks at a recent window (30 days): move the showcase's scored applications so the newest is now,
        # keeping their spacing, so the dashboard has data however long ago the image was built.
        newest = db.query(func.max(Prediction.created_at)).scalar()
        if newest is not None:
            shift = datetime.now(timezone.utc) - (newest if newest.tzinfo else newest.replace(tzinfo=timezone.utc))
            for row in db.query(Prediction).all():
                created = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=timezone.utc)
                row.created_at = created + shift
        users = db.query(User).all()
        for user in users:
            user.hashed_password = hash_password(secrets.token_urlsafe(32))
        db.commit()
        owner = os.environ.get("CREDIT_ADMIN_USERNAME", "").strip()
        owner_password = os.environ.get("CREDIT_ADMIN_PASSWORD", "")
        if owner and owner_password:
            if len(owner_password) < 12:
                print("[demo] CREDIT_ADMIN_PASSWORD must be at least 12 characters", file=sys.stderr)
                return 2
            UserRepository(db).create(
                username=owner,
                email=f"{owner}@owner.example.com",
                hashed_password=hash_password(owner_password),
                role="admin",
            )
            print(f"[demo] owner administrator {owner} ready", flush=True)
        else:
            print("[demo] CREDIT_ADMIN_USERNAME/CREDIT_ADMIN_PASSWORD not set: no administrator account", flush=True)
        print(f"[demo] locked {len(users)} showcase account(s); guest access on, registration off", flush=True)
    finally:
        db.close()

    # Behind the host's proxy: client addresses (rate limits, audit) come from X-Forwarded-For.
    os.environ.setdefault("FORWARDED_ALLOW_IPS", "*")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        os.environ.get("PORT", "8000"),
        "--workers",
        "1",
        "--proxy-headers",
        "--no-access-log",
    ]
    if os.name == "nt":  # local trial on Windows, where exec* mangles paths with spaces
        return subprocess.call(cmd)
    os.execvpe(cmd[0], cmd, os.environ)  # hand PID 1 to uvicorn so it receives SIGTERM
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
