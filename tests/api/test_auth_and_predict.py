"""API tests: auth flow, RBAC enforcement, single/batch prediction, explainability."""

from __future__ import annotations

SAMPLE_APPLICATION = {
    "customer_id": "TEST-API-001",
    "age": 35,
    "annual_income": 32000,
    "employment_length_years": 1.5,
    "loan_amount": 5000,
    "loan_term_months": 36,
    "interest_rate": 18.5,
    "credit_score": 590,
    "debt_to_income_ratio": 28.0,
    "num_open_accounts": 6,
    "num_credit_inquiries_6m": 3,
    "revolving_utilization": 0.7,
    "num_delinquencies_2yrs": 1,
    "months_since_last_delinquency": 10,
    "num_derogatory_marks": 0,
    "total_credit_limit": 8000,
    "savings_balance": 200,
    "checking_balance": 500,
    "num_dependents": 1,
    "years_at_residence": 2.0,
    "home_ownership": "RENT",
    "loan_purpose": "debt_consolidation",
    "employment_status": "employed",
    "education_level": "high_school",
    "marital_status": "single",
    "state": "CA",
}


_SELF_SERVICE_ROLES = {"viewer", "loan_officer"}


def _register_and_login(api_client, username="tester1", role="loan_officer") -> dict:
    """Returns Bearer-auth headers for a fresh user with the given role.

    Self-service roles (viewer, loan_officer) go through the real public
    HTTP register+login flow end to end -- exactly what most tests want to
    exercise. Privileged roles (admin, super_admin, risk_analyst, auditor,
    ml_engineer) can no longer self-register via the public endpoint (see
    `services/auth_service.py::AuthService.register`'s `SELF_SERVICE_ROLES`
    gate -- closing a real privilege-escalation hole where anyone could
    previously request `role="admin"` anonymously). For those, this helper
    seeds the user directly via the repository layer (the same DB the
    running app's `api_client` is wired to -- see `client.db_session_factory`
    in `tests/conftest.py`) and hand-mints a valid access token, exactly the
    way a real deployment seeds its first privileged accounts out-of-band
    via `scripts/create_admin.py` rather than over HTTP."""
    if role in _SELF_SERVICE_ROLES:
        api_client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": f"{username}@example.com",
                "password": "Supersecret1!",
                "role": role,
            },
        )
        resp = api_client.post("/api/v1/auth/login", json={"username": username, "password": "Supersecret1!"})
        assert resp.status_code == 200
        tokens = resp.json()
        return {"Authorization": f"Bearer {tokens['access_token']}"}

    from core.security import create_access_token, hash_password
    from repositories.user_repository import UserRepository

    db = api_client.db_session_factory()
    try:
        repo = UserRepository(db)
        user = repo.create(
            username=username,
            email=f"{username}@example.com",
            hashed_password=hash_password("Supersecret1!"),
            role=role,
        )
        token = create_access_token(user.id, user.role)
    finally:
        db.close()
    return {"Authorization": f"Bearer {token}"}


def test_health_endpoint_reports_champion(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_register_duplicate_username_conflicts(api_client):
    payload = {"username": "dupe1", "email": "dupe1@example.com", "password": "Supersecret1!", "role": "viewer"}
    r1 = api_client.post("/api/v1/auth/register", json=payload)
    r2 = api_client.post("/api/v1/auth/register", json=payload)
    assert r1.status_code == 201
    assert r2.status_code == 409


def test_login_wrong_password_returns_401(api_client):
    api_client.post(
        "/api/v1/auth/register",
        json={
            "username": "wrongpass",
            "email": "wp@example.com",
            "password": "Supersecret1!",
            "role": "viewer",
        },
    )
    resp = api_client.post("/api/v1/auth/login", json={"username": "wrongpass", "password": "nope"})
    assert resp.status_code == 401


def test_predict_requires_authentication(api_client):
    resp = api_client.post("/api/v1/predict", json=SAMPLE_APPLICATION)
    assert resp.status_code == 401


def test_predict_returns_valid_risk_assessment(api_client):
    headers = _register_and_login(api_client)
    resp = api_client.post("/api/v1/predict", json=SAMPLE_APPLICATION, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["predicted_probability"] <= 1.0
    assert body["decision"] in ("approved", "rejected")
    assert body["risk_band"] in ("very_low", "low", "medium", "high", "very_high")


def test_predict_rejects_invalid_credit_score(api_client):
    headers = _register_and_login(api_client)
    bad_application = {**SAMPLE_APPLICATION, "credit_score": 9999}  # out of [300, 850]
    resp = api_client.post("/api/v1/predict", json=bad_application, headers=headers)
    assert resp.status_code == 422


def test_batch_predict_returns_one_result_per_application(api_client):
    headers = _register_and_login(api_client, username="batch_tester")
    resp = api_client.post(
        "/api/v1/predict/batch",
        json={"applications": [SAMPLE_APPLICATION, SAMPLE_APPLICATION]},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["n_processed"] == 2


def test_viewer_role_cannot_score_predictions(api_client):
    headers = _register_and_login(api_client, username="viewer1", role="viewer")
    resp = api_client.post("/api/v1/predict", json=SAMPLE_APPLICATION, headers=headers)
    assert resp.status_code == 403


def test_admin_only_promote_endpoint_blocks_non_admin(api_client):
    headers = _register_and_login(api_client, username="officer1", role="loan_officer")
    resp = api_client.post("/api/v1/models/promote", json={"version": "test-v1"}, headers=headers)
    assert resp.status_code == 403


def test_explain_endpoint_returns_contributions(api_client):
    headers = _register_and_login(api_client, username="explainer1")
    resp = api_client.post("/api/v1/explain", json=SAMPLE_APPLICATION, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["contributions"]) > 0
    assert isinstance(body["plain_english"], str) and len(body["plain_english"]) > 0
