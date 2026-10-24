"""
Domain exception hierarchy.

Using typed exceptions (instead of bare `Exception` / `ValueError` everywhere)
lets the FastAPI exception handlers translate failures into the correct HTTP
status codes without string-matching error messages, and lets the pipeline
layers fail loudly and specifically. The service layer (`services/`) raises
these directly rather than `fastapi.HTTPException` -- keeping HTTP concerns
(status codes) entirely out of the service layer is what lets services stay
framework-agnostic and unit-testable without spinning up FastAPI.
"""

from __future__ import annotations


class CreditRiskPlatformError(Exception):
    """Base class for all platform-specific exceptions."""


# --- Data layer -------------------------------------------------------------
class DataValidationError(CreditRiskPlatformError):
    """Raised when incoming data fails schema, range, or quality checks."""


class DataDriftDetectedError(CreditRiskPlatformError):
    """Raised (or logged, depending on severity) when significant drift is detected."""


class DataLeakageError(CreditRiskPlatformError):
    """Raised when a feature is found to be statistically implausible (leakage)."""


# --- Model layer --------------------------------------------------------------
class ModelNotFoundError(CreditRiskPlatformError):
    """Raised when a requested model/version cannot be located in the registry."""


class ModelTrainingError(CreditRiskPlatformError):
    """Raised when a training run fails irrecoverably."""


class ModelLoadError(CreditRiskPlatformError):
    """Raised when a serialized model artifact cannot be deserialized."""


class InferenceError(CreditRiskPlatformError):
    """Raised when scoring fails after a model has been loaded successfully."""


# --- API / Security layer -----------------------------------------------------
class AuthenticationError(CreditRiskPlatformError):
    """Raised on invalid credentials or malformed/expired tokens."""


class AccountLockedError(AuthenticationError):
    """Raised when a login attempt is made against a temporarily locked
    account. Deliberately a subclass of AuthenticationError (not a sibling
    of it) so the existing @app.exception_handler(AuthenticationError) (401)
    keeps handling it via FastAPI/Starlette's MRO-based dispatch with zero
    changes to api/app.py -- only a more specific error message."""


class AuthorizationError(CreditRiskPlatformError):
    """Raised when an authenticated principal lacks the required role/permission."""


class RateLimitExceededError(CreditRiskPlatformError):
    """Raised when a client exceeds the configured request budget."""


class DuplicateResourceError(CreditRiskPlatformError):
    """Raised when attempting to create a resource that violates a uniqueness
    constraint the caller should have checked for (e.g. username already taken)."""


class ResourceNotFoundError(CreditRiskPlatformError):
    """Raised when a requested non-model resource (a user, an audit log
    entry, ...) does not exist. Kept distinct from ModelNotFoundError (which
    predates this and is specific to the model registry) so each
    exception's name stays accurate about what wasn't found."""


# --- Feature store -------------------------------------------------------------
class FeaturePipelineError(CreditRiskPlatformError):
    """Raised when a registered feature transformation fails or is misconfigured."""
