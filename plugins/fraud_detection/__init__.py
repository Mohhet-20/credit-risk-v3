"""
Fraud Detection plugin -- PLACEHOLDER ARCHITECTURE.

This platform does not ship a fraud-detection model or ruleset. This
module exists to define the STABLE INTERFACE `workflows/engine.py`'s
"fraud_check" step calls (`run_fraud_check(application: dict) -> dict`),
so that a real fraud-detection integration can be dropped in later without
touching the workflow engine, the scoring pipeline, or any router --
exactly the point of the plugin pattern described in `plugins/__init__.py`.

Why a placeholder instead of nothing: `docs/ROADMAP.md` and this module
both say the same thing explicitly rather than silently -- a workflow step
that always returns "not flagged" without disclosing that it isn't a real
check would be actively misleading in a compliance-sensitive pipeline. The
workflow engine's `fraud_check` step is disabled by default
(`SystemConfig` key `workflow.fraud_check_enabled=False`, see
`services/config_service.py::DEFAULT_CONFIG`) and, even when explicitly
enabled by an operator who understands what this currently does, always
returns a result whose `is_placeholder=True` flag makes that visible in
the workflow execution record.

A real implementation would typically call out to a rules/ML-based fraud
score (device fingerprinting, velocity checks, identity verification,
consortium data) and return `{"flagged": bool, "reason": str | None,
"score": float | None, "is_placeholder": False}` -- the same shape this
placeholder returns, so `WorkflowService` never needs to change when a
real implementation replaces this one.
"""

from __future__ import annotations


def run_fraud_check(application: dict) -> dict:
    """Always returns not-flagged. See module docstring -- this is
    intentionally NOT a real fraud check. `application` is accepted (and
    typed as the full applicant dict) to keep this function's signature
    identical to what a real implementation would need, even though this
    placeholder doesn't inspect it."""
    return {"flagged": False, "reason": None, "score": None, "is_placeholder": True}
