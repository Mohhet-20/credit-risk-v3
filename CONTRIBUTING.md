# Contributing

Thanks for considering a contribution to the Credit Risk Intelligence Platform. This
document covers the local workflow; see `docs/DEVELOPER_GUIDE.md` for a deeper tour of
the codebase's architecture and conventions.

## Getting set up

```bash
git clone <this-repo>
cd credit-risk-platform
python3 -m venv .venv && source .venv/bin/activate
make install-dev          # runtime deps + ruff + black + pytest-cov
cp .env.example .env      # adjust JWT_SECRET_KEY etc. for local dev
make db-upgrade           # apply Alembic migrations (creates the local SQLite DB)
make test                 # confirm everything passes before you start
```

## Workflow

1. **Open an issue or check `ROADMAP.md`** before starting substantial work, so effort
   isn't duplicated and the change fits the project's direction.
2. **Branch from `main`.** Small, focused branches review faster than one branch
   touching five subsystems.
3. **Write tests first or alongside the change**, not after. Every service method and
   endpoint added to this codebase has: a success-path test, an authorization-boundary
   test where relevant (wrong role → 403, no auth → 401), and a domain-exception test
   where applicable. See `docs/TESTING.md` for the full philosophy and directory
   layout.
4. **Run `make lint` and `make test` locally before opening a PR.** CI runs the same
   checks (`.github/workflows/ci.yml`); a red CI run on an otherwise-good PR just costs
   everyone a round trip.
5. **Update the relevant doc alongside the code.** A new endpoint updates
   `docs/API.md`; a new environment variable updates `.env.example` and
   `docs/OPERATIONS.md`; a new RBAC rule updates `docs/SECURITY.md` §2.2's
   authorization matrix. Docs that drift from the code are worse than no docs.
6. **Add a `CHANGELOG.md` entry** under an `[Unreleased]` heading (add one if it
   doesn't exist yet) describing the change from a user's perspective, not an
   implementation-detail perspective.

## Code style

- **Formatting**: `black` (120-char line length, see `pyproject.toml`). Run `make
  format` before committing — don't hand-format.
- **Linting**: `ruff`, config in `pyproject.toml`. A small, deliberate ignore list
  exists (`B008` for FastAPI's `Depends(...)` pattern, `UP038` for idiomatic
  `isinstance(x, (int, float))`) — see the inline comments in `pyproject.toml` before
  adding to it; don't silence a real finding by adding it to the ignore list.
- **Type hints**: this codebase uses them throughout (`from __future__ import
  annotations` at the top of nearly every module). New code should match.
- **Docstrings**: prefer explaining *why* a non-obvious decision was made over
  restating *what* the code does — the existing modules are full of "Design decision:
  ..." docstrings explaining tradeoffs; that's the house style, not decoration.
- **No new external LLM dependency without discussion.** `explainability/narrative.py`
  and `explainability/copilot.py` are deterministic and template-based by deliberate
  design (see their docstrings) — a PR introducing a live LLM call as the *default*
  path for an explanation should be discussed first, not merged as a drive-by change.

## Architecture boundaries to respect

- **Routers (`api/routers/`) parse, call a service, and shape the response** — no
  business logic and no direct repository/DB access in a router. See
  `docs/ARCHITECTURE.md`.
- **`api/dependencies.py` is the only place that wires concrete repositories into
  concrete services.** A new service should get a `get_x_service()` factory there, not
  be constructed ad hoc inside a router.
- **RBAC**: use `require_role("X")` for a simple "this tier or above" ordinal check,
  `require_roles(*roles_at_least("X"), "lateral_role")` when a lateral role
  (`auditor`, `ml_engineer`) should also be admitted. See `docs/DEVELOPER_GUIDE.md`'s
  "Adding a new authorization-gated endpoint" section.
- **Never bypass `configs/settings.py::Settings`** for reading configuration — no new
  bare `os.getenv()` calls scattered through the codebase.

## Database migrations

Model changes go through Alembic, not by hand-editing a table after the fact:

```bash
# 1. Edit database/models.py
# 2. Generate a migration from the diff:
make db-revision msg="add loan_purpose_other column to predictions"
# 3. Review the generated file in database/migrations/versions/ -- autogenerate
#    is a starting point, not a guarantee (it won't catch every constraint change)
# 4. Apply it locally to confirm it works:
make db-upgrade
```

## Reporting a security issue

Please do not open a public issue for a suspected security vulnerability. See
`docs/SECURITY.md` for the platform's current security posture and known scoped-out
areas — if you've found something beyond what's already disclosed there, report it
privately to the maintainers first.

## License

By contributing, you agree your contribution is licensed under this project's MIT
license (see `LICENSE`).
