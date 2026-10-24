# Developer Guide

## Local setup

```bash
python3 -m venv .venv && source .venv/bin/activate
make install-dev     # runtime deps + ruff/black/pytest-cov
cp .env.example .env
```

## Running things individually

```bash
make train            # python3 -m pipelines.run_training under the hood
make run-api           # uvicorn api.app:app --reload
make run-frontend       # streamlit run frontend/app.py
make mlflow-ui          # local MLflow tracking server on :5000
```

`make train` must be run at least once (or a model artifact + `champion.json` must already exist
under `models/artifacts/`) before the API's `/predict`, `/explain`, and `/analytics/*` endpoints
will return anything other than a 503 — see `api/dependencies.py::get_model_bundle`.

## Testing

```bash
make test           # pytest tests/ -v
make test-cov        # + coverage
```

Test layout mirrors the source tree:

- `tests/unit/` — preprocessing building blocks, training components, and the repository/service
  layer, all in isolation (see below)
- `tests/data/` — data-quality validators and drift detection
- `tests/integration/` — the full `FeaturePipeline` (including save/load round-trips) and the
  local `ModelRegistry` (register/promote/rollback)
- `tests/api/` — auth, RBAC, prediction, explainability, and analytics endpoints via FastAPI's
  `TestClient`, against an in-memory SQLite database and a small fixture-trained model

`tests/conftest.py` trains a single fast `LogisticRegression` on a 500-row synthetic dataset
rather than running the full Optuna-tuned 5-model comparison — fixture speed matters more than
model quality for testing pipeline *correctness*. All API tests use FastAPI's
`dependency_overrides` to inject the in-memory DB session and the fixture model bundle rather than
fighting the process-wide `get_settings()`/model-cache singletons. Because the override happens at
the lowest level (`get_db`, `get_model_registry`), it propagates automatically through every
repository and service built on top — `tests/unit/test_repositories.py` and
`tests/unit/test_services.py` construct repositories/services directly against the same
`db_session_factory` fixture with zero FastAPI involved.

A gotcha worth knowing if you extend `tests/conftest.py`: the in-memory SQLite engine must use
`poolclass=StaticPool`, or FastAPI's request-handling thread will open a second, empty in-memory
database instead of sharing the one `Base.metadata.create_all()` ran against.

## Code style

```bash
make lint       # ruff check . && black --check .
make format      # ruff check --fix . && black .
```

Configuration lives in `pyproject.toml` (120-char line length, Python 3.10 target).

## Adding a new API endpoint

The platform follows a strict Router -> Service -> Repository -> Database layering (see
[`ARCHITECTURE.md`](ARCHITECTURE.md#layered-architecture-api---service---repository---database)).
When adding an endpoint that touches the database:

1. **Repository.** If the query doesn't exist yet, add a method to the relevant class in
   `repositories/` (or a new `repositories/<aggregate>_repository.py` subclassing `BaseRepository`
   for a new aggregate). This is pure SQLAlchemy, no HTTP, no business rules.
2. **Service.** Add the business logic to the relevant class in `services/` (or a new
   `services/<domain>_service.py`), taking repositories as constructor arguments. Raise a domain
   exception from `core/exceptions.py` for anything that isn't the happy path — add a new
   exception class there (and a matching `@app.exception_handler` in `api/app.py`) if none of the
   existing ones fit.
3. **DI wiring.** Add `get_<name>_repository` / `get_<name>_service` provider functions to
   `api/dependencies.py` — this is the only file allowed to import both a repository and a
   service and wire them together.
4. **Schema.** Add the Pydantic request/response schema to `api/schemas.py`.
5. **Router.** Add the route to `api/routers/<router>.py` (or a new router file +
   `router = APIRouter(prefix="/...", tags=[...])`, mounted in `api/app.py` via
   `app.include_router(...)`). The handler body should be a few lines: parse -> call one service
   method -> return.
6. **Tests.** Add a service-level unit test in `tests/unit/test_services.py` (business logic,
   no HTTP) and, if the endpoint has RBAC/response-shape concerns, an API test in `tests/api/`
   using the `_register_and_login` helper pattern in `tests/api/test_auth_and_predict.py`.

Endpoints with no database write and no business rule beyond "call a pure-function package and
shape the response" (`/explain`, `/analytics/*`) skip the service layer and call the domain
package (`explainability/`, `analytics/`) directly from the router — introducing a service purely
for ceremony would violate YAGNI.

## Retraining / adding a model family

Model search spaces live in `configs/model_training.yaml`, not in Python — add a new entry under
`models:` with `enabled: true` and an Optuna `search_space`, then wire the estimator constructor
into `training/tuning.py`'s model factory. `core/sklearn_compat.py` documents a required monkeypatch
for XGBoost's `is_classifier()` compatibility with sklearn's tags API if you add another
XGBoost-family estimator.

## Known environment quirks

- `MLFLOW_ALLOW_FILE_STORE=true` must be set before importing `mlflow` (MLflow 3.x puts the
  plain filesystem tracking backend behind this flag). It's set in `tests/conftest.py`, the
  Makefile targets, and both Dockerfiles.
- Run pipeline scripts as modules (`python3 -m pipelines.run_training`), not as bare scripts
  (`python3 pipelines/run_training.py`) — the latter doesn't add the repo root to `sys.path`,
  so `configs`/`core`/etc. won't resolve.

## Adding a new authorization-gated endpoint

Two dependency patterns are available in `api/dependencies.py`, and picking the right one matters:

- `require_role("X")` — **ordinal**: admits `X` and every role above it on the `viewer < loan_officer < risk_analyst < admin < super_admin` ladder. Use this when "anyone at or above this seniority tier" is genuinely the right rule.
- `require_roles(*allowed)` — **explicit allow-list**: admits exactly the roles listed, nothing implied. Use this whenever a lateral role (`auditor`, `ml_engineer`) should be admitted alongside an ordinal tier — e.g. `require_roles(*roles_at_least("risk_analyst"), "auditor")`. `core/security.py::roles_at_least("X")` returns the ordinal set as a tuple so you can splice it into an explicit list without hand-maintaining the ladder.

See `docs/SECURITY.md` §2 for the full rationale on why lateral roles exist outside the ordinal ladder, and §2.2 for the current authorization matrix — update that table if you add a new gated endpoint.

## Bootstrapping a privileged account locally

`POST /auth/register` only accepts `viewer`/`loan_officer` (see `docs/SECURITY.md` §1.4). To get an admin account in a local dev database, run:

```bash
python3 -m scripts.create_admin
```

It prompts for username/email/password (via `getpass`, never a CLI argument) and writes directly through the repository layer — no HTTP round trip, no network exposure.

## Security & reliability additions this pass

- New middleware behavior, startup/shutdown lifecycle, and dependency-scanning CI job — see `docs/SECURITY.md` and `docs/OPERATIONS.md`.
- `api/app.py` now uses a `lifespan` async context manager (FastAPI's documented replacement for the deprecated `@app.on_event` decorator) — if you're adding new startup/shutdown behavior, add it there, not via a new `@app.on_event` handler.
- `configs/settings.py::Settings.validate_or_raise()` — if you add a new setting that would be unsafe with an insecure value in production, consider whether it belongs in `validate_production_secrets()`'s checklist.
