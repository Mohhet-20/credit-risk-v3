"""
Tests for `api/middleware.py`'s response headers, exercised through the
real ASGI stack (via the `api_client` fixture / FastAPI `TestClient`) rather
than unit-testing `SecurityHeadersMiddleware` in isolation -- what matters is
what an actual HTTP response carries, including header interactions with
CORS/rate-limit/correlation-ID middleware also registered on `app`.
"""

from __future__ import annotations


def test_health_response_carries_defensive_security_headers(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "geolocation=()" in response.headers["Permissions-Policy"]


def test_health_response_carries_content_security_policy(api_client):
    response = api_client.get("/health")
    csp = response.headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    # `img-src data:` is required for the base64-embedded matplotlib figures
    # in reporting/html_report.py -- verify the exception is scoped to img-src,
    # not a blanket allowance.
    assert "img-src 'self' data:" in csp
    assert "script-src 'self'" in csp


def test_debug_mode_omits_hsts_header(api_client):
    """The test app runs with `debug=True` (development default) -- HSTS
    must be absent, since it would be actively wrong advice for plain-HTTP
    local dev (see `SecurityHeadersMiddleware`'s docstring)."""
    response = api_client.get("/health")
    assert "Strict-Transport-Security" not in response.headers


def test_auth_endpoints_are_not_cached(api_client):
    response = api_client.post(
        "/api/v1/auth/register",
        json={
            "username": "cacheuser1",
            "email": "cacheuser1@example.com",
            "password": "Supersecret1!",
            "role": "viewer",
        },
    )
    assert response.headers.get("Cache-Control") == "no-store"


def test_non_auth_endpoint_has_no_cache_control_override(api_client):
    response = api_client.get("/health")
    assert response.headers.get("Cache-Control") != "no-store"


def test_correlation_id_is_generated_and_echoed(api_client):
    response = api_client.get("/health")
    assert "X-Correlation-ID" in response.headers
    assert len(response.headers["X-Correlation-ID"]) > 0


def test_correlation_id_is_propagated_from_request_header(api_client):
    supplied = "test-correlation-id-12345"
    response = api_client.get("/health", headers={"X-Correlation-ID": supplied})
    assert response.headers["X-Correlation-ID"] == supplied


def test_process_time_header_present(api_client):
    response = api_client.get("/health")
    assert "X-Process-Time-Ms" in response.headers
    assert float(response.headers["X-Process-Time-Ms"]) >= 0
