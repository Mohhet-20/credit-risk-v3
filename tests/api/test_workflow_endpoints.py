"""API tests for the Enterprise Workflow Engine (`/workflow/*`) -- the full
governed pipeline (validation -> data quality -> fraud check -> AI
prediction -> business rules -> manual review -> decision -> report ->
notification -> audit log), executed end to end against the same
in-memory DB and fixture model bundle every other API test uses."""

from __future__ import annotations

from tests.api.test_auth_and_predict import SAMPLE_APPLICATION, _register_and_login


def test_workflow_definition_lists_standard_steps(api_client):
    headers = _register_and_login(api_client, username="wf_def", role="loan_officer")
    resp = api_client.get("/api/v1/workflow/definition", headers=headers)
    assert resp.status_code == 200
    steps = resp.json()["steps"]
    assert steps[0] == "validation"
    assert steps[-1] == "audit_logging"
    assert "business_rules" in steps


def test_execute_workflow_requires_authentication(api_client):
    resp = api_client.post("/api/v1/workflow/execute", json=SAMPLE_APPLICATION)
    assert resp.status_code == 401


def test_execute_workflow_end_to_end_produces_a_decision(api_client):
    headers = _register_and_login(api_client, username="wf_exec", role="loan_officer")
    resp = api_client.post("/api/v1/workflow/execute", json=SAMPLE_APPLICATION, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("completed", "failed")
    assert body["final_decision"] in ("approved", "rejected", "manual_review")
    assert body["prediction"]["prediction_id"]
    assert isinstance(body["steps"], list)
    step_names = [s["name"] for s in body["steps"]]
    # fraud_check defaults to disabled (see DEFAULT_CONFIG), so it must be
    # recorded as skipped, not silently omitted.
    assert "fraud_check" in step_names
    fraud_step = next(s for s in body["steps"] if s["name"] == "fraud_check")
    assert fraud_step["status"] == "skipped"


def test_execute_workflow_aborts_when_fraud_check_flags_application(api_client, monkeypatch):
    """Regression test: the fraud_check step must abort the workflow when
    flagged, the same way validation/data_quality_check do. It previously
    ran with `non_fatal=True`, which recorded the step as 'failed' but
    swallowed the exception and let scoring/decisioning proceed anyway."""
    import plugins.fraud_detection as fraud_detection

    monkeypatch.setattr(
        fraud_detection,
        "run_fraud_check",
        lambda application: {"flagged": True, "reason": "test-forced-flag", "score": 0.99, "is_placeholder": False},
    )

    admin_headers = _register_and_login(api_client, username="wf_fraud_admin", role="admin")
    enable_resp = api_client.put(
        "/api/v1/config/workflow.fraud_check_enabled",
        headers=admin_headers,
        json={"value": True, "category": "workflow"},
    )
    assert enable_resp.status_code == 200

    exec_headers = _register_and_login(api_client, username="wf_fraud_exec", role="loan_officer")
    resp = api_client.post("/api/v1/workflow/execute", json=SAMPLE_APPLICATION, headers=exec_headers)

    assert resp.status_code == 400
    assert "fraud" in resp.json()["detail"].lower()


def test_execute_workflow_rejects_invalid_application(api_client):
    headers = _register_and_login(api_client, username="wf_invalid", role="loan_officer")
    bad_application = {**SAMPLE_APPLICATION, "credit_score": 9999}
    resp = api_client.post("/api/v1/workflow/execute", json=bad_application, headers=headers)
    assert resp.status_code == 422  # Pydantic Field bounds reject this before the workflow even starts


def test_execute_workflow_persists_execution_and_is_retrievable(api_client):
    headers = _register_and_login(api_client, username="wf_persist", role="loan_officer")
    exec_resp = api_client.post("/api/v1/workflow/execute", json=SAMPLE_APPLICATION, headers=headers)
    execution_id = exec_resp.json()["execution_id"]

    reader_headers = _register_and_login(api_client, username="wf_reader", role="risk_analyst")
    get_resp = api_client.get(f"/api/v1/workflow/executions/{execution_id}", headers=reader_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == execution_id


def test_get_workflow_execution_404_for_unknown_id(api_client):
    headers = _register_and_login(api_client, username="wf_404", role="risk_analyst")
    resp = api_client.get("/api/v1/workflow/executions/does-not-exist", headers=headers)
    assert resp.status_code == 404


def test_list_workflow_executions_admits_auditor(api_client):
    exec_headers = _register_and_login(api_client, username="wf_lister_exec", role="loan_officer")
    api_client.post("/api/v1/workflow/execute", json=SAMPLE_APPLICATION, headers=exec_headers)

    auditor_headers = _register_and_login(api_client, username="wf_lister_auditor", role="auditor")
    resp = api_client.get("/api/v1/workflow/executions", headers=auditor_headers)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_list_workflow_executions_blocks_loan_officer(api_client):
    headers = _register_and_login(api_client, username="wf_list_lo", role="loan_officer")
    resp = api_client.get("/api/v1/workflow/executions", headers=headers)
    assert resp.status_code == 403


def test_workflow_stats_returns_counts(api_client):
    exec_headers = _register_and_login(api_client, username="wf_stats_exec", role="loan_officer")
    api_client.post("/api/v1/workflow/execute", json=SAMPLE_APPLICATION, headers=exec_headers)

    stats_headers = _register_and_login(api_client, username="wf_stats_reader", role="risk_analyst")
    resp = api_client.get("/api/v1/workflow/stats", headers=stats_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert "completed" in body and "failed" in body


def test_business_rule_override_beats_model_decision(api_client):
    """A rule with a broad, always-true condition (matches whatever the
    fixture model bundle's raw features contain) must override the model's
    own decision -- proving the workflow's rule-evaluation step actually
    consults the rule engine rather than just recording an empty result."""
    rule_headers = _register_and_login(api_client, username="wf_rule_setup", role="risk_analyst")
    create_resp = api_client.post(
        "/api/v1/rules",
        json={
            "name": "force manual review",
            "rule_type": "manual_review",
            "condition": {"field": "age", "operator": "gte", "value": 0},  # always true for any real applicant
            "priority": 1,
        },
        headers=rule_headers,
    )
    assert create_resp.status_code == 201

    exec_headers = _register_and_login(api_client, username="wf_rule_exec", role="loan_officer")
    resp = api_client.post("/api/v1/workflow/execute", json=SAMPLE_APPLICATION, headers=exec_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_decision"] == "manual_review"
    assert body["needs_manual_review"] is True
    assert body["rule_evaluation"]["matched_rule_type"] == "manual_review"
