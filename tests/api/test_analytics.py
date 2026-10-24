"""API tests for the bonus analytics endpoints: segmentation, anomaly
detection, loan recommendation, and the feature-importance timeline."""

from __future__ import annotations

from tests.api.test_auth_and_predict import SAMPLE_APPLICATION, _register_and_login


def test_segment_endpoint_returns_a_labeled_segment(api_client):
    headers = _register_and_login(api_client, username="segment_tester")
    resp = api_client.post("/api/v1/analytics/segment", json=SAMPLE_APPLICATION, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["segment_id"], int)
    assert isinstance(body["label"], str) and len(body["label"]) > 0


def test_anomaly_endpoint_returns_a_score(api_client):
    headers = _register_and_login(api_client, username="anomaly_tester")
    resp = api_client.post("/api/v1/analytics/anomaly", json=SAMPLE_APPLICATION, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["anomaly_score"] <= 1.0
    assert isinstance(body["is_anomaly"], bool)
    assert len(body["top_deviant_features"]) > 0


def test_recommend_endpoint_returns_loan_options(api_client):
    headers = _register_and_login(api_client, username="recommend_tester")
    resp = api_client.post("/api/v1/analytics/recommend", json=SAMPLE_APPLICATION, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "recommendations" in body
    assert body["n_options_found"] == len(body["recommendations"])
    for option in body["recommendations"]:
        assert option["approved"] is True


def test_feature_importance_timeline_requires_risk_analyst_role(api_client):
    headers = _register_and_login(api_client, username="loan_officer_1", role="loan_officer")
    resp = api_client.get("/api/v1/analytics/feature-importance-timeline", headers=headers)
    assert resp.status_code == 403


def test_feature_importance_timeline_returns_data_for_risk_analyst(api_client):
    headers = _register_and_login(api_client, username="analyst_1", role="risk_analyst")
    resp = api_client.get("/api/v1/analytics/feature-importance-timeline", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "timeline" in body
    assert len(body["timeline"]) >= 1
    assert body["timeline"][0]["version"] == "test-v1"
