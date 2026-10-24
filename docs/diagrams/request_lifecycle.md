# Request Lifecycle & Security Layers

How a single authenticated API request actually flows through the stack, in the order
each layer is applied — useful alongside `docs/SECURITY.md` when reasoning about
*where* a given control is enforced.

```mermaid
flowchart TD
    C[Client: Streamlit frontend / external API consumer] -->|HTTPS request + Bearer token| MW1

    subgraph Middleware["ASGI middleware stack (api/middleware.py, api/app.py)"]
        direction TB
        MW1[CorrelationIdMiddleware\nassigns/propagates X-Correlation-ID] --> MW2
        MW2[SecurityHeadersMiddleware\nCSP, X-Frame-Options, HSTS, Cache-Control] --> MW3
        MW3[CORSMiddleware] --> MW4
        MW4[SlowAPIMiddleware\nglobal + route-level rate limits] --> MW5
        MW5[request_timing_and_logging\nstructured JSON log line per request]
    end

    MW5 --> DEP

    subgraph DepChain["FastAPI dependency chain (api/dependencies.py)"]
        direction TB
        DEP[get_current_user\ndecode JWT, verify type=access,\nload user, check is_active] --> RBAC
        RBAC["require_role() / require_roles()\nordinal ladder OR explicit lateral allow-list"] --> SVC
        SVC[Service factory\ne.g. get_scoring_service]
    end

    RBAC -->|403 if role check fails| ERR1[JSONResponse 403]
    DEP -->|401 if token invalid/expired/wrong type| ERR2[JSONResponse 401]

    SVC --> ROUTER

    subgraph AppLayer["Router -> Service -> Repository (the only layer with business logic)"]
        direction TB
        ROUTER["Router handler\nparse request, call service, shape response\n(no business logic here)"] --> SERVICE
        SERVICE["Service\nbusiness rules, orchestrates repositories,\nraises typed domain exceptions"] --> REPO
        REPO["Repository\nSQLAlchemy ORM queries only,\nnever raw SQL string formatting"] --> DB[(PostgreSQL / SQLite)]
    end

    SERVICE -->|domain exception, e.g. AccountLockedError| HANDLERS
    HANDLERS["Typed exception handlers (api/app.py)\nmap CreditRiskPlatformError subclasses\nto the correct HTTP status"] --> RESP

    ROUTER --> RESP[Response]
    RESP -->|response headers added on the way out| MW2
    MW2 --> C2[Client receives response\n+ X-Correlation-ID, X-Process-Time-Ms,\nsecurity headers]
```

## Why this ordering matters

- **Correlation ID is assigned before anything else runs** so every subsequent log
  line — including a rejected-at-the-door 401/403 — carries it, which matters for
  reconstructing an incident from logs alone.
- **Rate limiting runs before authentication is even checked** for most endpoints
  (global default), so a credential-stuffing burst against `/auth/login` is throttled
  before it reaches `AuthService.login()`'s account-lockout logic at all — two
  independent layers of brute-force defense (see `docs/SECURITY.md` §1.5).
- **RBAC runs strictly after authentication**, never combined into one check — a
  request with a missing/invalid token always fails as 401 (authentication), never
  403 (authorization), which matters for not leaking "this resource exists but you
  can't see it" information to a fully unauthenticated caller.
- **The router layer never talks to the database directly** — every arrow from
  `ROUTER` goes through `SERVICE` then `REPO`. This is what makes `services/*.py`
  independently unit-testable (see `docs/TESTING.md`) without spinning up FastAPI at
  all.
