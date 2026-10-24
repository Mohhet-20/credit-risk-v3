"""API tests: scorecard points + adverse-action reasons on /predict and
/explain, the /auth/me profile endpoint, and /models/validation-report."""

from __future__ import annotations

import json

import pytest

from explainability import explainer_cache
from tests.api.test_auth_and_predict import SAMPLE_APPLICATION, _register_and_login

HIGH_RISK = {
    **SAMPLE_APPLICATION,
    "customer_id": "TEST-DECLINE-001",
    "annual_income": 18000,
    "loan_amount": 30000,
    "credit_score": 510,
    "debt_to_income_ratio": 55.0,
    "revolving_utilization": 0.98,
    "num_delinquencies_2yrs": 5,
    "months_since_last_delinquency": 2,
    "num_derogatory_marks": 3,
    "num_credit_inquiries_6m": 9,
    "employment_status": "unemployed",
    "employment_length_years": 0.0,
}

LOW_RISK = {
    **SAMPLE_APPLICATION,
    "customer_id": "TEST-APPROVE-001",
    "annual_income": 240000,
    "loan_amount": 2000,
    "credit_score": 830,
    "debt_to_income_ratio": 4.0,
    "revolving_utilization": 0.03,
    "num_delinquencies_2yrs": 0,
    "months_since_last_delinquency": 999,
    "num_derogatory_marks": 0,
    "num_credit_inquiries_6m": 0,
    "savings_balance": 90000,
    "home_ownership": "OWN",
}


@pytest.fixture(autouse=True)
def _background_from_fixture(monkeypatch, raw_dataset):
    """Serve SHAP background rows from the session fixture instead of disk."""
    sample = raw_dataset.drop(columns=["default_flag"]).sample(60, random_state=0)
    monkeypatch.setattr(explainer_cache, "background_sample", lambda version, n=100: sample)
    explainer_cache._explainers.clear()
    yield
    explainer_cache._explainers.clear()


def test_declined_prediction_carries_reasons_and_score(api_client):
    headers = _register_and_login(api_client, username="officer_decline")
    resp = api_client.post("/api/v1/predict", json=HIGH_RISK, headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["decision"] == "rejected"
    assert 300 <= body["credit_score"] <= 850
    reasons = body["adverse_action_reasons"]
    assert 1 <= len(reasons) <= 4
    assert [r["rank"] for r in reasons] == list(range(1, len(reasons) + 1))
    assert all(r["contribution"] > 0 and r["reason"] for r in reasons)
    assert all(r["code"] != "R99" or r["features"] for r in reasons)


def test_approved_prediction_has_no_reasons_and_higher_score(api_client):
    headers = _register_and_login(api_client, username="officer_approve")
    approved = api_client.post("/api/v1/predict", json=LOW_RISK, headers=headers).json()
    declined = api_client.post("/api/v1/predict", json=HIGH_RISK, headers=headers).json()
    assert approved["decision"] == "approved"
    assert approved["adverse_action_reasons"] == []
    assert approved["credit_score"] > declined["credit_score"]


def test_decline_reason_codes_are_written_to_the_audit_trail(api_client):
    from repositories.audit_log_repository import AuditLogRepository

    headers = _register_and_login(api_client, username="officer_audit")
    body = api_client.post("/api/v1/predict", json=HIGH_RISK, headers=headers).json()
    with api_client.db_session_factory() as db:
        entries = [e for e in AuditLogRepository(db).list_audit_logs(limit=50) if e.action == "predict"]
    details = entries[0].details if isinstance(entries[0].details, dict) else json.loads(entries[0].details)
    assert details["decision"] == "rejected"
    assert details["reason_codes"] == [r["code"] for r in body["adverse_action_reasons"]]


def test_explain_endpoint_returns_reasons(api_client):
    headers = _register_and_login(api_client, username="officer_explain")
    resp = api_client.post("/api/v1/explain", json=HIGH_RISK, headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["adverse_action_reasons"]


def test_protected_attributes_do_not_change_the_decision(api_client):
    headers = _register_and_login(api_client, username="officer_fair")
    base = api_client.post("/api/v1/predict", json=HIGH_RISK, headers=headers).json()
    altered = api_client.post(
        "/api/v1/predict",
        json={**HIGH_RISK, "age": 72, "marital_status": "married", "state": "TX"},
        headers=headers,
    ).json()
    assert altered["predicted_probability"] == pytest.approx(base["predicted_probability"])


def test_auth_me_returns_the_callers_profile(api_client):
    headers = _register_and_login(api_client, username="me_user", role="viewer")
    resp = api_client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "me_user"
    assert body["role"] == "viewer"
    assert "hashed_password" not in body


def test_auth_me_requires_authentication(api_client):
    assert api_client.get("/api/v1/auth/me").status_code == 401


def test_validation_report_404_until_trained_then_served(api_client, tmp_path, monkeypatch):
    from configs.settings import get_settings

    headers = _register_and_login(api_client, username="viewer_report", role="viewer")
    monkeypatch.setattr(get_settings(), "reports_dir", tmp_path)
    assert api_client.get("/api/v1/models/validation-report", headers=headers).status_code == 404

    (tmp_path / "model_validation.json").write_text(json.dumps({"model_name": "lightgbm", "oot_metrics": {"ks": 0.5}}))
    resp = api_client.get("/api/v1/models/validation-report", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["oot_metrics"]["ks"] == 0.5
