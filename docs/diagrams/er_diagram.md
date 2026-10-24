# Entity-Relationship Diagram

Generated from `database/models.py` (the single source of truth for schema — see
`docs/OPERATIONS.md`'s "Database migrations" section for how `database/migrations/`
keeps this in sync via Alembic).

```mermaid
erDiagram
    USERS ||--o{ PREDICTIONS : "requests (nullable, API-initiated predictions have none)"
    USERS ||--o{ AUDIT_LOGS : "performs"

    USERS {
        string id PK
        string username UK
        string email UK
        string hashed_password
        string role "viewer|loan_officer|risk_analyst|admin|super_admin|auditor|ml_engineer"
        bool is_active
        datetime created_at
        int failed_login_attempts
        datetime locked_until "nullable"
        datetime last_login_at "nullable"
        int token_version "refresh-token rotation counter"
        string mfa_secret "nullable, TOTP secret"
        bool mfa_enabled
        text mfa_backup_codes "nullable, JSON list of bcrypt-hashed codes"
    }

    PREDICTIONS {
        string id PK
        string customer_id "nullable, indexed"
        json input_features
        float predicted_probability
        string risk_band
        string decision "approved|rejected|manual_review"
        float decision_threshold
        string model_version
        string model_name
        float latency_ms "nullable"
        string request_source "api|batch|streaming"
        string requested_by FK "nullable -> users.id"
        datetime created_at "indexed"
    }

    AUDIT_LOGS {
        string id PK
        string user_id FK "nullable -> users.id"
        string action
        string resource "nullable"
        json details "nullable"
        string ip_address "nullable"
        datetime created_at "indexed"
    }

    MODEL_VERSIONS {
        string id PK
        string version UK
        string model_name
        string stage "None|Staging|Production|Archived"
        json metrics "nullable"
        float decision_threshold "nullable"
        datetime registered_at
    }

    EXPERIMENT_RUNS {
        string id PK
        string run_name
        string mlflow_run_id "nullable"
        string model_name
        json params "nullable"
        json metrics "nullable"
        datetime created_at
    }

    DATA_QUALITY_REPORTS {
        string id PK
        string report_type "schema|drift|quality"
        json summary
        bool passed
        datetime created_at "indexed"
    }
```

## Notes

- **`MODEL_VERSIONS` and `EXPERIMENT_RUNS` have no foreign keys into `USERS`.** These
  mirror the local `ModelRegistry` / MLflow tracking store into Postgres for fast
  dashboard queries — the filesystem/MLflow store remains the source of truth for the
  artifacts themselves, and "who promoted this version" is recorded in `AUDIT_LOGS`
  (`action="model.promote"`, `resource=<version>`) rather than a column here, since an
  audit trail belongs in the audit log, not duplicated onto the domain table.
- **`PREDICTIONS.requested_by` is nullable** because not every prediction is tied to an
  authenticated user session — batch/streaming ingestion paths can score applications
  without a `current_user` in scope.
- **All primary keys are UUIDv4 strings**, not auto-incrementing integers — see
  `database/models.py::_uuid()`. This avoids leaking a row-count/creation-order signal
  through predictable IDs (a resource-ID enumeration/information-disclosure concern
  independent of the RBAC layer) and keeps IDs generation-agnostic between SQLite
  (local dev) and PostgreSQL (production).
