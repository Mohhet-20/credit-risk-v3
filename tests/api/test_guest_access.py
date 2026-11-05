"""Public-demo controls: guest sessions, the registration switch, and the SQLite opt-in for a single-instance demo."""

import pytest

from configs.settings import Settings, get_settings
from core.security import hash_password
from repositories.user_repository import UserRepository
from tests.api.test_auth_and_predict import SAMPLE_APPLICATION


@pytest.fixture
def demo_env(monkeypatch):
    def apply(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()

    yield apply
    get_settings.cache_clear()


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_guest_access_is_off_by_default(api_client):
    assert api_client.post("/api/v1/auth/guest").status_code == 404


def test_guest_is_a_risk_analyst_who_can_score_and_explain(api_client, demo_env):
    demo_env(GUEST_ACCESS_ENABLED="true", GUEST_USERNAME="guest-ra")
    res = api_client.post("/api/v1/auth/guest")
    assert res.status_code == 200, res.text
    headers = _bearer(res.json()["access_token"])
    me = api_client.get("/api/v1/auth/me", headers=headers).json()
    assert me["username"] == "guest-ra" and me["role"] == "risk_analyst"
    scored = api_client.post("/api/v1/predict", json=SAMPLE_APPLICATION, headers=headers)
    assert scored.status_code == 200 and scored.json()["decision"] in {"approved", "rejected", "manual_review"}
    assert api_client.post("/api/v1/explain", json=SAMPLE_APPLICATION, headers=headers).status_code == 200
    # The same account on the next visit.
    again = api_client.post("/api/v1/auth/guest").json()
    assert api_client.get("/api/v1/auth/me", headers=_bearer(again["access_token"])).json()["id"] == me["id"]


def test_guest_cannot_administer(api_client, demo_env):
    demo_env(GUEST_ACCESS_ENABLED="true", GUEST_USERNAME="guest-admin-check")
    headers = _bearer(api_client.post("/api/v1/auth/guest").json()["access_token"])
    assert api_client.get("/api/v1/admin/users", headers=headers).status_code == 403


def test_a_more_powerful_account_is_never_handed_out_as_the_guest(api_client, demo_env):
    db = api_client.db_session_factory()
    try:
        UserRepository(db).create(
            username="guest-root",
            email="root@example.com",
            hashed_password=hash_password("Irrelevant-Password-1"),
            role="admin",
        )
    finally:
        db.close()
    demo_env(GUEST_ACCESS_ENABLED="true", GUEST_USERNAME="guest-root")
    assert api_client.post("/api/v1/auth/guest").status_code == 403


def test_registration_can_be_turned_off(api_client, demo_env):
    demo_env(REGISTRATION_ENABLED="false")
    res = api_client.post(
        "/api/v1/auth/register",
        json={
            "username": "newbie",
            "email": "newbie@example.com",
            "password": "Sufficiently-Long-Pass-9!",
            "role": "viewer",
        },
    )
    assert res.status_code == 404


def test_sqlite_in_production_needs_an_explicit_single_instance_opt_in():
    base = {
        "app_env": "production",
        "jwt_secret_key": "k" * 16 + "Zq8-vT3_pW1xYc7nR5mL",
        "database_url": "sqlite:///x.db",
    }
    assert any("SQLite" in p for p in Settings(**base).validate_production_secrets())
    assert not any(
        "SQLite" in p for p in Settings(**base, allow_sqlite_in_production=True).validate_production_secrets()
    )
