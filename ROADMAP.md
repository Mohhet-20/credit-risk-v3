# Roadmap

This roadmap is deliberately honest about what's built versus what's a real next step —
every item here traces back to a gap explicitly disclosed in `docs/SECURITY.md` §7,
`docs/OPERATIONS.md`, or the design-decision comments throughout the codebase, not
aspirational feature marketing.

## Near term (natural next increments)

- **Less-discriminatory-alternative search.** The fair-lending review flags disparities; the
  next step is to search for a model with comparable accuracy and a smaller disparity before
  accepting one.
- **Champion/challenger traffic split.** Score a share of live traffic with the challenger in
  shadow mode before promotion, instead of an atomic swap.
- **Reject inference and a defined performance window** (e.g. 12-month default) once trained on
  real bureau and performance data.
- **MFA step in the web console.** Accounts with MFA enabled currently need the Streamlit app or
  the API to complete login.

- **Confirm `pip-audit` runs clean in real CI.** The dependency-scanning job
  (`.github/workflows/ci.yml`) has not been executed successfully in this development
  sandbox (network-restricted); verify and tighten `continue-on-error` once it has a
  documented run history and a team-agreed remediation SLA for findings.
- **Real network-level load testing.** `tests/api/test_performance.py` is a fast,
  useful in-process smoke test, not a substitute for k6/Locust/Gatling against a
  deployed instance under sustained concurrent traffic.
- **Field-level export for the remaining report types** (drift, evaluation, data
  quality). CSV/Excel export currently covers prediction history, audit logs, and
  model comparison — the three most-requested tabular exports; evaluation/drift/
  data-quality reports remain HTML+PDF only.

## Medium term

- **Email-based account flows.** No email verification on registration, no
  password-reset-via-email. Today, password resets go through an admin or
  current-password-plus-MFA; adding self-service email flows needs an email-delivery
  integration (e.g. SES/SendGrid), which is a real external dependency this platform
  has so far deliberately avoided taking on.
- **A real LLM-backed Copilot mode**, additive alongside the current deterministic
  template-based Copilot (`explainability/copilot.py`), for free-form natural-language
  Q&A ("why did applicant X get declined but applicant Y got approved?"). Requires
  provisioning an LLM API key and accepting non-deterministic output for a
  conversational layer — the existing structured summaries would remain the
  deterministic, auditable default.
- **Kubernetes-native autoscaling and multi-region deployment guidance.** Current
  Kubernetes manifests (`deployment/kubernetes/`) cover a single-region deployment;
  horizontal pod autoscaling policies and multi-region active/active guidance are not
  yet documented.
## Long term / exploratory

- **Field-level PII encryption at rest**, if a stricter regulatory regime (or a
  multi-tenant deployment) requires it beyond the current RBAC + standard database
  access control model.
- **Streamlit accessibility beyond the current contrast/labeling audit.** Deeper
  screen-reader support for the embedded Plotly charts is a Streamlit-platform-level
  limitation, not something fixable from application code — would require either a
  Streamlit-native accessibility improvement upstream, or a migration to a different
  frontend framework entirely.
- **A managed feature store** (e.g. Feast) if the platform ever needs point-in-time
  correct feature serving across multiple models sharing overlapping features — the
  current `feature_store/` module is a well-tested in-process feature pipeline, not a
  distributed feature store, and that's the right scope for a single-model-family
  platform today.

## Explicitly not planned

- Replacing the deterministic SHAP/LIME/template-based explanations with an opaque
  LLM-generated explanation as the *default* explanation path — for a regulated
  credit-risk product, reproducible and auditable explanations are a requirement, not
  a limitation to engineer away (see `explainability/narrative.py` and
  `explainability/copilot.py`'s docstrings).
