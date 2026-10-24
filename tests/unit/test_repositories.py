"""
Unit tests for the repository layer, exercised directly against an
in-memory SQLite session with no FastAPI machinery involved -- this is the
testability payoff of the Repository Pattern described in
`repositories/__init__.py`.
"""

from __future__ import annotations

from repositories.audit_log_repository import AuditLogRepository
from repositories.data_quality_repository import DataQualityReportRepository
from repositories.model_version_repository import ModelVersionRepository
from repositories.prediction_repository import PredictionRepository
from repositories.user_repository import UserRepository


def test_user_repository_create_and_get_by_username(db_session_factory):
    db = db_session_factory()
    repo = UserRepository(db)

    created = repo.create(username="alice", email="alice@example.com", hashed_password="hashed", role="viewer")
    assert created.id is not None

    found = repo.get_by_username("alice")
    assert found is not None
    assert found.email == "alice@example.com"
    assert repo.get_by_username("nobody") is None


def test_prediction_repository_create_and_filter_by_customer(db_session_factory):
    db = db_session_factory()
    repo = PredictionRepository(db)

    for cid in ["CUST-A", "CUST-A", "CUST-B"]:
        repo.create(
            customer_id=cid,
            input_features={"age": 30},
            predicted_probability=0.1,
            risk_band="low",
            decision="approved",
            decision_threshold=0.2,
            model_version="v1",
            model_name="logreg",
            latency_ms=5.0,
            request_source="api",
            requested_by=None,
        )

    all_records = repo.list_predictions(limit=100)
    assert len(all_records) == 3

    filtered = repo.list_predictions(limit=100, customer_id="CUST-A")
    assert len(filtered) == 2
    assert all(r.customer_id == "CUST-A" for r in filtered)


def test_prediction_repository_stats_last_n_days(db_session_factory):
    db = db_session_factory()
    repo = PredictionRepository(db)
    repo.create(
        customer_id="X",
        input_features={},
        predicted_probability=0.4,
        risk_band="medium",
        decision="rejected",
        decision_threshold=0.3,
        model_version="v1",
        model_name="logreg",
        latency_ms=12.0,
        request_source="api",
        requested_by=None,
    )
    stats = repo.stats_last_n_days(days=30)
    assert stats["n_predictions"] == 1
    assert stats["avg_predicted_probability"] == 0.4


def test_audit_log_repository_log_and_list(db_session_factory):
    db = db_session_factory()
    repo = AuditLogRepository(db)
    repo.log_action(user_id="u1", action="login", resource="user:u1")
    logs = repo.list_audit_logs(limit=10)
    assert len(logs) == 1
    assert logs[0].action == "login"


def test_model_version_repository_upsert_creates_then_updates(db_session_factory):
    db = db_session_factory()
    repo = ModelVersionRepository(db)

    created = repo.upsert(version="v1", model_name="logreg", stage="Staging")
    assert created.stage == "Staging"

    updated = repo.upsert(version="v1", model_name="logreg", stage="Production", decision_threshold=0.25)
    assert updated.id == created.id  # same row, upserted not duplicated
    assert updated.stage == "Production"
    assert updated.decision_threshold == 0.25

    versions = repo.list_model_versions()
    assert len(versions) == 1


def test_data_quality_repository_save_and_list_reports(db_session_factory):
    db = db_session_factory()
    repo = DataQualityReportRepository(db)

    saved = repo.save_report(report_type="quality_report", summary={"n_rows": 10}, passed=True)
    assert saved.id is not None
    assert saved.passed is True

    reports = repo.list_reports(report_type="quality_report", limit=10)
    assert len(reports) == 1
    assert reports[0].summary["n_rows"] == 10


def test_data_quality_repository_list_reports_filters_by_type(db_session_factory):
    db = db_session_factory()
    repo = DataQualityReportRepository(db)
    repo.save_report(report_type="quality_report", summary={}, passed=True)
    repo.save_report(report_type="other_type", summary={}, passed=True)

    only_quality = repo.list_reports(report_type="quality_report", limit=10)
    assert len(only_quality) == 1

    everything = repo.list_reports(limit=10)
    assert len(everything) == 2


def test_data_quality_repository_list_reports_page_returns_total_and_ordering(db_session_factory):
    db = db_session_factory()
    repo = DataQualityReportRepository(db)
    for i in range(5):
        repo.save_report(report_type="quality_report", summary={"run": i}, passed=True)

    items, total = repo.list_reports_page(report_type="quality_report", limit=2, offset=0)
    assert total == 5
    assert len(items) == 2
    # Most recently created report first (descending by created_at).
    assert items[0].summary["run"] == 4

    items_page_2, total_2 = repo.list_reports_page(report_type="quality_report", limit=2, offset=2)
    assert total_2 == 5
    assert len(items_page_2) == 2
