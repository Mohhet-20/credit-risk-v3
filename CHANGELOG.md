# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries are grouped by the
development pass that introduced them, since this platform was built as a sequence of
deliberate, reviewed increments rather than a single commit.

## [3.1.0] — Hosted demo

### Added
- **Hosted demo on free hosting** (`render.yaml`, `deploy/demo/Dockerfile`, `web/vercel.json`): serves the validated
  champion from `deploy/demo/champion/`; 300 out-of-time applications scored at build time (`scripts/demo_build.py`);
  fresh showcase, locked accounts and current timestamps on every start (`scripts/demo_start.py`).
- **Guest sessions.** `POST /auth/guest` (off unless `GUEST_ACCESS_ENABLED`) signs visitors in as a shared,
  rate-limited `risk_analyst`, never a more powerful account; the web console offers *Explore the live demo* when built
  with `NEXT_PUBLIC_GUEST_ACCESS=true`.
- `REGISTRATION_ENABLED` switch for self-service sign-up, and `ALLOW_SQLITE_IN_PRODUCTION` as an explicit opt-in for a
  single-instance deployment.

### Fixed
- Docker images left out `reports/model_validation.json`, so `GET /models/validation-report` had nothing to serve
  outside a bind-mounted checkout; `.dockerignore` now admits the validation report.
- The web console treats `NEXT_PUBLIC_API_BASE_URL=/` as same-origin (behind a proxy) instead of a host.
- Lint/format drift that failed CI's `ruff`/`black --check` (`core/security.py`, `frontend/app.py`,
  `reporting/model_comparison_report.py`).

## [3.0.0] — Model-risk validation, fair lending and the web console

### Added
- **Out-of-time validation.** Training splits by `application_date` (60% train / 20%
  validation / 20% OOT). Calibration, champion selection and the decision threshold use the
  validation window only; the OOT window is reported once. Duplicate applications are removed
  before splitting.
- **Credit validation metrics** (`training/credit_metrics.py`): Gini, KS, decile gains/lift and
  bootstrap 95% CIs. Each run writes `reports/MODEL_VALIDATION.md` and `model_validation.json`,
  served at `GET /models/validation-report`.
- **Adverse-action reason codes** (`explainability/adverse_action.py`): every declined `/predict`
  and `/explain` response carries up to four ECOA / Reg B principal reasons, also written to the
  audit log.
- **Fair-lending review** (`analytics/fairness.py`): adverse impact ratio and four-fifths-rule
  check by age band and marital status, with per-group calibration and AUC.
- **Scorecard points** (`core/scorecard.py`): PDO scaling, 600 = 50:1 odds, +20 doubles.
- **Monotonic constraints** on XGBoost, LightGBM and CatBoost for business-sensible risk
  directions, verified end to end through the feature pipeline (score and utilisation bands are
  now descriptive only, so they cannot bypass the constraints).
- **Parsimony rule:** ensembles must beat the best single model by `ensemble.min_improvement`.
- **Web console** (`web/`): scoring page with decline reasons, model-validation page, `/auth/me`
  login flow; Dockerfile, compose service and CI job (lint, type-check, build).
- `GET /auth/me`; per-version SHAP explainer cache with the background sample stored beside
  each model artifact.

### Changed
- **Protected and proxy attributes removed from the model.** `age`, `marital_status` and `state`
  are dropped before any pipeline fitting and kept only for fairness testing.
- Synthetic dataset grows from 500 to 50,000 applications and gains a mild credit-cycle drift.
- Model Card and Data Card describe the fair-lending review that now runs.

### Fixed
- **Calibration discarded the trained model.** `CalibratedClassifierCV(cv=5)` refit clones on the
  calibration split alone; all models now use prefit calibration on the full training fit.
- **Test-set leakage:** the threshold and the champion were both chosen on the test split.
- `make train` omitted the required `--stage` and had no data on a fresh clone; `make
  docker-train` failed for the same reason. Training now generates the dataset when missing.
- `make create-admin` omitted required arguments.
- Ruff's `exclude` replaced its defaults (it linted `.venv`); Black was failing on 161 files.
- SHAP explainers were rebuilt on every request.
- KS used `scipy.stats.ks_2samp`, whose exact p-value made bootstrapped KS hang; it is now
  computed directly as max(TPR - FPR).

## [2.2.0] — Elite enterprise enhancements

### Added
- **Business Rule Engine** (`/rules/*`): admin/risk-analyst-editable approval,
  rejection, and manual-review rules with a closed JSON condition grammar (never a
  free-text expression evaluated at runtime — a rule is data, not code), priority-
  ordered first-match-wins evaluation, dry-run simulation (`POST /rules/simulate`),
  and version-bumped edits with prior-state snapshots in the audit log (see
  `rules/engine.py`, `services/rule_service.py`).
- **Enterprise Workflow Engine** (`/workflow/*`): the full governed pipeline —
  validation, data-quality check, fraud check (placeholder architecture, see
  `plugins/fraud_detection/`), AI prediction, business rules, manual-review gate,
  decision, report generation, notification, audit logging — recorded step-by-step
  and persisted as a queryable `WorkflowExecution`. Additive alongside the existing
  `POST /predict`, not a replacement (see `services/workflow_service.py`).
- **Enterprise Configuration Center** (`/config/*`): a single `SystemConfig` table
  for feature flags, workflow toggles, and thresholds (e.g. the manual-review
  probability band), with sane built-in defaults so the platform behaves correctly
  before an admin ever touches it (see `services/config_service.py`).
- **AI Governance** (`/reports/model-card`, `/reports/data-card`,
  `/reports/compliance-dashboard`): a Model Card (intended use, evaluation metrics,
  and an explicit, undisguised statement of what fairness/bias auditing was NOT
  performed) and a Data Card (provenance, schema, synthetic-data disclosure), plus a
  cross-subsystem Compliance Dashboard aggregating model-governance, rule, workflow,
  and data-quality status from data those subsystems already track.
- **Copilot governance summary** (`/copilot/summary/governance`): a narrative layer
  over the Compliance Dashboard, and a new governance section in
  `/copilot/executive-briefing`.
- **Modular plugin pattern** (`plugins/`): a real, minimal `plugins/fraud_detection/`
  placeholder with a stable interface the Workflow Engine already calls — established
  as a pattern via one genuine extension point, not a set of empty stub directories.
- Reviewed the existing synthetic data generator (`data/generate_synthetic_data.py`)
  against this pass's "Synthetic Data Generator" requirement — already supports
  configurable dataset size, realistic causally-structured features, and injected
  missingness/outliers/label noise/duplicates; no changes were needed.

### Notes
- Deep visual/UX polish requested in this pass (spacing, animations, command palette,
  etc.) has the same Streamlit-architecture caveat documented in the 2.1.0 entry below
  and was not re-attempted here.

## [2.1.0] — Enterprise completion pass

### Added
- **AI Copilot** (`/copilot/*`): explain a prediction in plain English with recommended
  actions, summarize model performance / monitoring / data quality, and a combined
  executive briefing endpoint. Deterministic, template-based NLG over existing
  structured data — no external LLM dependency (see `explainability/copilot.py`).
- **CSV/Excel export** (`/reports/export/*`): prediction history, audit logs, and model
  comparison, alongside the existing HTML and PDF report formats (`reporting/export.py`).
- **Real Alembic migration scaffold** (`database/migrations/`, `alembic.ini`): the
  `alembic` dependency had been pinned since an earlier pass but never wired to an
  actual migration; `make db-upgrade` / `db-downgrade` / `db-revision` now work against
  whichever database `DATABASE_URL` points at.
- `docs/CHANGELOG.md`... `CHANGELOG.md`, `ROADMAP.md`, `CONTRIBUTING.md`, `LICENSE`.

### Removed
- Unused pinned dependencies: `seaborn` (matplotlib is used directly everywhere;
  seaborn was never imported), `imbalanced-learn` (class imbalance is handled via
  `class_weight` and threshold optimization, not resampling; never imported).
- Empty, unreferenced `frontend/components/` directory (dead scaffolding).

## [2.0.0] — Enterprise security, testing & reliability pass

### Added
- Password policy enforcement, 7-role RBAC (ordinal ladder + lateral `auditor`/
  `ml_engineer` roles), admin-issued user creation, out-of-band admin bootstrap CLI
  (`scripts/create_admin.py`), refresh-token rotation, real TOTP-based MFA with
  backup codes.
- Content-Security-Policy and other hardened response headers; `Cache-Control:
  no-store` on auth endpoints; secrets validation that fails startup on an insecure
  production configuration; retry-with-backoff database initialization; graceful
  shutdown; `lifespan`-based startup (replacing the deprecated `on_event` hooks).
- Weekly + per-push dependency vulnerability scanning in CI (`pip-audit`).
- 260+ tests across unit, data, infra, integration, and API layers.
- `docs/SECURITY.md`, `docs/TESTING.md`, `docs/TROUBLESHOOTING.md`, `docs/OPERATIONS.md`.

### Fixed
- **Security**: closed a privilege-escalation hole where public registration
  accepted an arbitrary `role` (including `admin`). Closed a reflected-XSS vector in
  HTML reports where an unescaped applicant ID was interpolated into report titles.
  Closed a fully-unauthenticated model-metadata exposure on `GET /models/versions`
  and `GET /models/comparison`.

## [1.5.0] — MLOps & cloud platform pass

### Added
- Multi-stage Docker builds for API, frontend, and ML services; Docker Compose with
  profiles, network isolation, and prod/dev/test environment layering; Redis-backed
  distributed cache (with in-memory fallback); CI pipeline building all three images.

## [1.4.0] — Data quality platform pass

### Added
- Expanded schema/business-rule validation; a dataset profiling module; a composite
  0–100 data-quality score; persisted history via `DataQualityService`; a
  `/data-quality` API surface and HTML report.

## [1.3.0] — AI/explainability platform pass

### Added
- LIME explainer as a comparison method alongside SHAP; a shared HTML report
  scaffold; Evaluation, Model Comparison, Drift, Performance Summary, and
  Explainability report generators; a `/reports` router.

## [1.2.0] — Enterprise backend pass

### Added
- Repository + service layer refactor (routers no longer touch the DB directly);
  correlation-ID and security-headers middleware; response caching and pagination;
  account lockout after repeated failed logins; admin and health routers; rate
  limiting.

## [1.0.0] — Initial platform

### Added
- Tuned gradient-boosting model zoo (XGBoost/LightGBM/CatBoost) with Optuna
  hyperparameter search, calibration, and cost-sensitive threshold optimization;
  MLflow experiment tracking and a local model registry with promote/rollback; SHAP
  explainability; a JWT-secured FastAPI backend; a Streamlit dashboard; PostgreSQL
  persistence; a full test suite and initial Docker/CI setup.
