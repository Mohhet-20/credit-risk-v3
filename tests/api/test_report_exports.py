"""API tests for the CSV/Excel tabular export endpoints on the reports
router (`/reports/export/*`)."""

from __future__ import annotations

import io

import openpyxl

from tests.api.test_auth_and_predict import SAMPLE_APPLICATION, _register_and_login


def test_export_predictions_csv(api_client):
    headers = _register_and_login(api_client, username="export1", role="loan_officer")
    api_client.post("/api/v1/predict", json=SAMPLE_APPLICATION, headers=headers)

    ra_headers = _register_and_login(api_client, username="export1b", role="risk_analyst")
    resp = api_client.get("/api/v1/reports/export/predictions?format=csv", headers=ra_headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]
    assert b"predicted_probability" in resp.content


def test_export_predictions_xlsx(api_client):
    headers = _register_and_login(api_client, username="export2", role="loan_officer")
    api_client.post("/api/v1/predict", json=SAMPLE_APPLICATION, headers=headers)

    ra_headers = _register_and_login(api_client, username="export2b", role="risk_analyst")
    resp = api_client.get("/api/v1/reports/export/predictions?format=xlsx", headers=ra_headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/vnd.openxmlformats")
    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    ws = wb.active
    header_row = [c.value for c in ws[1]]
    assert "predicted_probability" in header_row


def test_export_predictions_requires_risk_analyst_or_above(api_client):
    headers = _register_and_login(api_client, username="export3", role="loan_officer")
    resp = api_client.get("/api/v1/reports/export/predictions", headers=headers)
    assert resp.status_code == 403


def test_export_audit_logs_requires_admin_or_auditor(api_client):
    headers = _register_and_login(api_client, username="export4", role="risk_analyst")
    resp = api_client.get("/api/v1/reports/export/audit-logs", headers=headers)
    assert resp.status_code == 403

    admin_headers = _register_and_login(api_client, username="export4b", role="admin")
    resp2 = api_client.get("/api/v1/reports/export/audit-logs?format=csv", headers=admin_headers)
    assert resp2.status_code == 200
    assert resp2.headers["content-type"].startswith("text/csv")


def test_export_audit_logs_admits_auditor(api_client):
    headers = _register_and_login(api_client, username="export5", role="auditor")
    resp = api_client.get("/api/v1/reports/export/audit-logs", headers=headers)
    assert resp.status_code == 200


def test_export_model_comparison_csv(api_client):
    headers = _register_and_login(api_client, username="export6", role="risk_analyst")
    resp = api_client.get("/api/v1/reports/export/model-comparison?format=csv", headers=headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")


def test_export_defaults_to_csv_for_unknown_format(api_client):
    headers = _register_and_login(api_client, username="export7", role="risk_analyst")
    resp = api_client.get("/api/v1/reports/export/model-comparison?format=bogus", headers=headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
