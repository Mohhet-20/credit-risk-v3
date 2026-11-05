"""Build the hosted public demo's showcase (run once, at image build time, by deploy/demo/Dockerfile).

The demo serves the exact champion behind the README's numbers and reports/MODEL_VALIDATION.md
(`deploy/demo/champion/`, ~180 KB, produced by `make train`), so nothing is retrained on the free host.
This script, run through the real API in-process as an administrator created for the build only (random
password, locked again by scripts/demo_start.py), leaves under CREDIT_DEMO_ROOT (default /tmp/credit-demo):

* ``artifacts/`` - the model registry with the champion pointer;
* ``app.db`` - 300 applications from the out-of-time window of the seeded synthetic dataset, scored through
  POST /predict, so the dashboard's monitoring, score distribution and history have real data.
"""

from __future__ import annotations

import os
import secrets
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = Path(os.environ.get("CREDIT_DEMO_ROOT", "/tmp/credit-demo"))
CHAMPION = ROOT / "deploy" / "demo" / "champion"
N_SCORED = 300


def main() -> int:
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)
    shutil.copytree(CHAMPION, DEMO_ROOT / "artifacts")
    os.environ.update(
        {
            "APP_ENV": "production",
            "DEBUG": "false",
            "ALLOW_SQLITE_IN_PRODUCTION": "true",
            "JWT_SECRET_KEY": secrets.token_urlsafe(48),  # build-only; tokens issued here are never used again
            "DATABASE_URL": f"sqlite:///{(DEMO_ROOT / 'app.db').as_posix()}",
            "MODEL_ARTIFACT_DIR": str(DEMO_ROOT / "artifacts"),
            "MLFLOW_ALLOW_FILE_STORE": "true",
            "MLFLOW_TRACKING_URI": str(DEMO_ROOT / "mlruns"),
            "RATE_LIMIT_PER_MINUTE": "100000",
        }
    )
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    from fastapi.testclient import TestClient

    from api.app import app
    from api.schemas import CreditApplicationRequest
    from core.rate_limit import limiter
    from core.security import create_access_token, hash_password
    from data.generate_synthetic_data import generate_credit_dataset
    from database.session import SessionLocal, init_db
    from repositories.user_repository import UserRepository

    init_db()
    db = SessionLocal()
    try:
        builder = UserRepository(db).create(
            username="showcase-builder",
            email="builder@showcase.example.com",
            hashed_password=hash_password(secrets.token_urlsafe(32)),
            role="admin",
        )
        token = create_access_token(builder.id, builder.role)
    finally:
        db.close()

    # The latest applications by date are in the out-of-time window the model never saw.
    # The generator deliberately includes messy intake rows (missing values, outliers) for the data-quality checks;
    # the API rejects those at validation, so the showcase scores the latest N_SCORED applications that pass it.
    from pydantic import ValidationError

    df = generate_credit_dataset().sort_values("application_date", ascending=False)
    fields = list(CreditApplicationRequest.model_fields)
    applications, skipped = [], 0
    for row in df.to_dict(orient="records"):
        try:
            body = {k: row[k] for k in fields if k in row}
            body["customer_id"] = str(row.get("customer_id") or "")
            applications.append(CreditApplicationRequest(**body).model_dump(mode="json"))
        except (ValidationError, ValueError, TypeError):
            skipped += 1
        if len(applications) == N_SCORED:
            break
    limiter.enabled = False
    decisions: dict[str, int] = {}
    with TestClient(app) as c:
        c.headers["Authorization"] = f"Bearer {token}"
        for body in reversed(applications):  # oldest first, so history reads in time order
            res = c.post("/api/v1/predict", json=body)
            if res.status_code != 200:
                raise SystemExit(f"[demo-build] POST /predict -> {res.status_code}: {res.text[:300]}")
            decision = res.json()["decision"]
            decisions[decision] = decisions.get(decision, 0) + 1
        if c.get("/api/v1/models/validation-report").status_code != 200:
            raise SystemExit("[demo-build] the validation report is not served")
    print(
        f"[demo-build] scored {N_SCORED} out-of-time applications ({skipped} messy rows skipped): {decisions}",
        flush=True,
    )
    print(f"[demo-build] showcase ready in {DEMO_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
