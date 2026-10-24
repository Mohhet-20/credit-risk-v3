"""API tests for the data-quality endpoints (`/data-quality/run`,
`/data-quality/latest`, `/data-quality/history`) and the corresponding
`/reports/data-quality` downloadable HTML report."""

from __future__ import annotations

from tests.api.test_auth_and_predict import _register_and_login


def test_run_data_quality_check_requires_authentication(api_client):
    resp = api_client.post("/api/v1/data-quality/run")
    assert resp.status_code == 401


def test_run_data_quality_check_blocks_viewer_role(api_client):
    headers = _register_and_login(api_client, username="dq_viewer", role="viewer")
    resp = api_client.post("/api/v1/data-quality/run", headers=headers)
    assert resp.status_code == 403


def test_run_data_quality_check_persists_and_returns_result(api_client):
    headers = _register_and_login(api_client, username="dq_runner", role="risk_analyst")
    resp = api_client.post("/api/v1/data-quality/run", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["report_type"] == "quality_report"
    assert body["dataset_label"] == "reference"
    assert "validation" in body and "profile" in body and "score" in body


def test_latest_data_quality_report_404_before_any_run(api_client):
    headers = _register_and_login(api_client, username="dq_latest_empty", role="risk_analyst")
    resp = api_client.get("/api/v1/data-quality/latest", headers=headers)
    assert resp.status_code == 404


def test_latest_data_quality_report_returns_most_recent_run(api_client):
    headers = _register_and_login(api_client, username="dq_latest", role="risk_analyst")
    api_client.post("/api/v1/data-quality/run", headers=headers)
    resp = api_client.get("/api/v1/data-quality/latest", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["report_type"] == "quality_report"


def test_data_quality_history_is_paginated(api_client):
    headers = _register_and_login(api_client, username="dq_history", role="risk_analyst")
    api_client.post("/api/v1/data-quality/run", headers=headers)
    api_client.post("/api/v1/data-quality/run", headers=headers)

    resp = api_client.get("/api/v1/data-quality/history?page=1&size=1", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 2
    assert len(body["items"]) == 1


def test_data_quality_report_html_404_before_any_run(api_client):
    headers = _register_and_login(api_client, username="dq_report_empty", role="risk_analyst")
    resp = api_client.get("/api/v1/reports/data-quality", headers=headers)
    assert resp.status_code == 404


def test_data_quality_report_html_renders_after_run(api_client):
    headers = _register_and_login(api_client, username="dq_report_tester", role="risk_analyst")
    run_resp = api_client.post("/api/v1/data-quality/run", headers=headers)
    assert run_resp.status_code == 200

    resp = api_client.get("/api/v1/reports/data-quality", headers=headers)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Data Quality Report" in resp.text
    assert "Validation Issues" in resp.text


def test_data_quality_report_html_blocks_loan_officer(api_client):
    headers = _register_and_login(api_client, username="dq_report_lo", role="loan_officer")
    resp = api_client.get("/api/v1/reports/data-quality", headers=headers)
    assert resp.status_code == 403
