"""
Thin HTTP client wrapping the FastAPI backend, with the JWT access token
cached in Streamlit's `session_state`.

Design decision: the frontend talks to the platform exclusively over HTTP
(never imports `training`/`models`/`explainability` directly), the same way
any other API consumer would. This keeps the "premium dashboard" honest as a
real client of the versioned API rather than a shortcut that would silently
diverge from what external integrators actually get.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_PREFIX = "/api/v1"


def _headers() -> dict:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _url(path: str) -> str:
    return f"{API_BASE_URL}{API_PREFIX}{path}"


class ApiError(Exception):
    pass


def _request(method: str, url: str, **kwargs) -> requests.Response:
    """Issues the HTTP request, converting transport-level failures (backend
    unreachable, DNS failure, timeout) into `ApiError` so every call site's
    existing `except api_client.ApiError` also covers a backend outage,
    instead of only HTTP-status errors raised by `_handle()`."""
    try:
        return requests.request(method, url, **kwargs)
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Could not reach the API ({url}): {exc}") from exc


def _handle(resp: requests.Response) -> Any:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise ApiError(f"{resp.status_code}: {detail}")
    if resp.content:
        return resp.json()
    return None


def health() -> dict:
    return _handle(_request("GET", f"{API_BASE_URL}/health", timeout=10))


def login(username: str, password: str) -> dict:
    """Calls `POST /auth/login`. If the account has MFA enabled, the backend
    returns `mfa_required=True` with no tokens -- the caller must then collect
    a TOTP/backup code and call `verify_mfa_login()` with the returned
    `mfa_challenge_token` before any token is stored."""
    resp = _request("POST", _url("/auth/login"), json={"username": username, "password": password}, timeout=10)
    data = _handle(resp)
    if data.get("mfa_required"):
        return data
    st.session_state["access_token"] = data["access_token"]
    st.session_state["refresh_token"] = data["refresh_token"]
    return data


def verify_mfa_login(mfa_challenge_token: str, code: str) -> dict:
    """Exchanges an MFA challenge token + TOTP/backup code (from `login()`'s
    `mfa_required` response) for real tokens via `POST /auth/mfa/verify-login`."""
    resp = _request(
        "POST",
        _url("/auth/mfa/verify-login"),
        json={"mfa_challenge_token": mfa_challenge_token, "code": code},
        timeout=10,
    )
    data = _handle(resp)
    st.session_state["access_token"] = data["access_token"]
    st.session_state["refresh_token"] = data["refresh_token"]
    return data


def register(username: str, email: str, password: str, role: str = "loan_officer") -> dict:
    resp = _request(
        "POST",
        _url("/auth/register"),
        json={"username": username, "email": email, "password": password, "role": role},
        timeout=10,
    )
    return _handle(resp)


def is_authenticated() -> bool:
    return bool(st.session_state.get("access_token"))


def logout() -> None:
    st.session_state.pop("access_token", None)
    st.session_state.pop("refresh_token", None)


def predict(application: dict) -> dict:
    return _handle(_request("POST", _url("/predict"), json=application, headers=_headers(), timeout=30))


def predict_batch(applications: list[dict]) -> dict:
    return _handle(
        _request("POST", _url("/predict/batch"), json={"applications": applications}, headers=_headers(), timeout=120)
    )


def explain(application: dict) -> dict:
    return _handle(_request("POST", _url("/explain"), json=application, headers=_headers(), timeout=30))


def counterfactual(application: dict) -> dict:
    return _handle(_request("POST", _url("/explain/counterfactual"), json=application, headers=_headers(), timeout=30))


def download_report(application: dict) -> bytes:
    resp = _request("POST", _url("/explain/report"), json=application, headers=_headers(), timeout=30)
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise ApiError(f"{resp.status_code}: {detail}")
    return resp.content


def model_versions() -> list[dict]:
    return _handle(_request("GET", _url("/models/versions"), headers=_headers(), timeout=10))


def model_comparison() -> list[dict]:
    return _handle(_request("GET", _url("/models/comparison"), headers=_headers(), timeout=10))


def promote_model(version: str) -> dict:
    return _handle(_request("POST", _url("/models/promote"), json={"version": version}, headers=_headers(), timeout=10))


def monitoring_stats(days: int = 30) -> dict:
    return _handle(_request("GET", _url("/monitoring/stats"), params={"days": days}, headers=_headers(), timeout=10))


def prediction_distribution(days: int = 30) -> dict:
    return _handle(
        _request(
            "GET",
            _url("/monitoring/prediction-distribution"),
            params={"days": days},
            headers=_headers(),
            timeout=10,
        )
    )


def drift_report() -> dict:
    return _handle(_request("GET", _url("/monitoring/drift"), headers=_headers(), timeout=30))


def latency_stats() -> dict:
    return _handle(_request("GET", _url("/monitoring/latency"), headers=_headers(), timeout=10))


def segment(application: dict) -> dict:
    return _handle(_request("POST", _url("/analytics/segment"), json=application, headers=_headers(), timeout=30))


def anomaly(application: dict) -> dict:
    return _handle(_request("POST", _url("/analytics/anomaly"), json=application, headers=_headers(), timeout=30))


def recommend(application: dict) -> dict:
    return _handle(_request("POST", _url("/analytics/recommend"), json=application, headers=_headers(), timeout=30))


def feature_importance_timeline() -> dict:
    return _handle(_request("GET", _url("/analytics/feature-importance-timeline"), headers=_headers(), timeout=15))
