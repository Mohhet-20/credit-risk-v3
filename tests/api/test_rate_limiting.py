"""
API test for the stricter per-route rate limit on `/auth/login`.

The `api_client` fixture disables the global limiter for every other test
(see `tests/conftest.py`) so unrelated tests aren't affected by shared
in-process limiter state. This test explicitly re-enables it to verify the
limiter itself actually engages and returns 429 once the per-route budget
(10/minute, `api/routers/auth.py`) is exceeded.
"""

from __future__ import annotations

from core.rate_limit import limiter


def test_login_rate_limit_returns_429_after_budget_exceeded(api_client):
    api_client.post(
        "/api/v1/auth/register",
        json={
            "username": "ratelimited1",
            "email": "rl1@example.com",
            "password": "Supersecret1!",
            "role": "viewer",
        },
    )

    limiter.enabled = True
    try:
        statuses = []
        for _ in range(15):
            resp = api_client.post(
                "/api/v1/auth/login",
                json={"username": "ratelimited1", "password": "wrong-password"},
            )
            statuses.append(resp.status_code)
        assert 429 in statuses
    finally:
        limiter.enabled = False
