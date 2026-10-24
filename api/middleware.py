"""
ASGI middleware: request correlation IDs and security response headers.

Design decision: implemented as standalone `BaseHTTPMiddleware` subclasses
in their own module (not inline in `api/app.py`) so they're independently
unit-testable and `api/app.py` stays focused on wiring — the same reasoning
behind keeping `api/dependencies.py` a pure composition root.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from configs.settings import get_settings
from core.logging import set_correlation_id

CORRELATION_ID_HEADER = "X-Correlation-ID"

RequestResponseEndpoint = Callable[[Request], Awaitable[Response]]

# Endpoints whose responses must never be cached by a shared/browser cache --
# tokens and MFA challenge material can leak across users on a misconfigured
# shared cache or persist in browser history/back-forward cache otherwise.
_NO_STORE_PATH_FRAGMENTS = ("/auth/",)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Accepts an inbound `X-Correlation-ID` header (so an upstream gateway
    or caller can supply one for cross-service tracing) or generates a fresh
    UUID4 otherwise. The ID is stored in a `contextvars.ContextVar`
    (`core/logging.py`) for the lifetime of the request, so every log line
    emitted anywhere in the call stack — router, service, repository,
    background task — is automatically tagged with it via
    `CorrelationIdFilter`, with zero changes needed to business-logic code.
    The ID is also echoed back as a response header so a caller can
    correlate their own logs against ours."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or str(uuid.uuid4())
        set_correlation_id(correlation_id)
        request.state.correlation_id = correlation_id
        try:
            response = await call_next(request)
        finally:
            set_correlation_id(None)
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds standard defensive response headers to every response.

    `Strict-Transport-Security` is only added outside debug mode: it tells
    browsers to refuse plain HTTP for this host for the given max-age,
    which is actively wrong advice for a local/dev deployment served over
    plain HTTP and would lock developers out of `http://localhost`.

    `Content-Security-Policy` matters here specifically because this API
    isn't purely a JSON API — `api/routers/reports.py` serves raw
    `HTMLResponse` pages (model comparison, evaluation, drift, data-quality,
    and per-applicant explainability reports) that a browser will render
    directly. A restrictive CSP is defense-in-depth against XSS in those
    pages even though `reporting/html_report.py::render_report()` already
    escapes the only known user-influenced field (see that module's
    docstring on the `customer_id` reflected-XSS fix). `'unsafe-inline'` is
    required for `style-src` and `img-src data:` because the reports embed
    inline `<style>` (no external stylesheet) and base64 `data:` URIs for
    matplotlib figures (no static asset server) by design — see
    `reporting/html_report.py`'s module docstring. No `script-src` beyond
    `'self'` is granted since none of these pages ship any JavaScript.

    `Cache-Control: no-store` is added on `/auth/*` paths specifically
    (login, refresh, MFA endpoints) so access/refresh tokens and MFA
    challenge tokens returned in JSON response bodies are never written to
    a shared HTTP cache or a browser's disk cache."""

    _CSP = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = self._CSP
        if not get_settings().debug:
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        if any(fragment in request.url.path for fragment in _NO_STORE_PATH_FRAGMENTS):
            response.headers["Cache-Control"] = "no-store"
        return response
