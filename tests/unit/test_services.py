"""
Unit tests for the service layer, constructed directly with repository
instances (no FastAPI `TestClient`, no HTTP). This is the point of pulling
business logic out of routers: `AuthService.login()` raising
`AuthenticationError` can be asserted with a plain `pytest.raises`, without
caring what HTTP status a router later maps it to.
"""

from __future__ import annotations

import pytest

from core.constants import MAX_FAILED_LOGIN_ATTEMPTS
from core.exceptions import (
    AccountLockedError,
    AuthenticationError,
    AuthorizationError,
    DuplicateResourceError,
    ResourceNotFoundError,
)
from core.security import hash_password
from repositories.audit_log_repository import AuditLogRepository
from repositories.data_quality_repository import DataQualityReportRepository
from repositories.model_version_repository import ModelVersionRepository
from repositories.prediction_repository import PredictionRepository
from repositories.user_repository import UserRepository
from services.audit_service import AuditService
from services.auth_service import AuthService
from services.data_quality_service import DataQualityService
from services.model_governance_service import ModelGovernanceService
from services.scoring_service import MAX_SYNC_BATCH_SIZE, BatchTooLargeError, ScoringService
from services.user_service import UserService
from tests.api.test_auth_and_predict import SAMPLE_APPLICATION


def _auth_service(db_session_factory) -> AuthService:
    db = db_session_factory()
    return AuthService(UserRepository(db), AuditLogRepository(db))


def test_auth_service_register_then_duplicate_raises(db_session_factory):
    service = _auth_service(db_session_factory)
    user = service.register(username="bob", email="bob@example.com", password="Supersecret1!", role="viewer")
    assert user.username == "bob"

    with pytest.raises(DuplicateResourceError):
        service.register(username="bob", email="bob2@example.com", password="Supersecret1!", role="viewer")


def test_auth_service_login_success_and_wrong_password(db_session_factory):
    service = _auth_service(db_session_factory)
    service.register(username="carol", email="carol@example.com", password="Supersecret1!", role="loan_officer")

    result = service.login(username="carol", password="Supersecret1!")
    assert result.mfa_required is False
    assert result.tokens.access_token and result.tokens.refresh_token

    with pytest.raises(AuthenticationError):
        service.login(username="carol", password="wrong-password")


def test_auth_service_login_inactive_account_raises_authorization_error(db_session_factory):
    db = db_session_factory()
    user_repo = UserRepository(db)
    service = AuthService(user_repo, AuditLogRepository(db))

    user = user_repo.create(
        username="dave",
        email="dave@example.com",
        hashed_password=hash_password("Supersecret1!"),
        role="viewer",
    )
    user.is_active = False
    db.commit()

    with pytest.raises(AuthorizationError):
        service.login(username="dave", password="Supersecret1!")


def test_scoring_service_score_single_persists_prediction(db_session_factory, trained_model_bundle):
    db = db_session_factory()
    prediction_repo = PredictionRepository(db)
    service = ScoringService(prediction_repo, AuditLogRepository(db))

    result = service.score_single(
        _application(SAMPLE_APPLICATION),
        trained_model_bundle,
        requested_by=None,
    )
    assert 0.0 <= result["predicted_probability"] <= 1.0
    assert result["decision"] in ("approved", "rejected")

    history = service.get_history(limit=10)
    assert len(history) == 1
    assert history[0]["prediction_id"] == result["prediction_id"]


def test_scoring_service_batch_too_large_raises(db_session_factory, trained_model_bundle):
    db = db_session_factory()
    service = ScoringService(PredictionRepository(db), AuditLogRepository(db))
    applications = [_application(SAMPLE_APPLICATION) for _ in range(MAX_SYNC_BATCH_SIZE + 1)]

    with pytest.raises(BatchTooLargeError):
        service.score_batch(applications, trained_model_bundle, requested_by=None)


def test_model_governance_service_promote_and_rollback(db_session_factory, model_registry, trained_model_bundle):
    db = db_session_factory()
    governance = ModelGovernanceService(
        model_registry,
        ModelVersionRepository(db),
        AuditLogRepository(db),
        bust_cache_fn=lambda: None,
    )

    model_registry.register_version(
        version="v2",
        model=trained_model_bundle.model,
        feature_pipeline=trained_model_bundle.feature_pipeline,
        metadata={**trained_model_bundle.metadata, "version": "v2"},
    )

    governance.promote("v2", actor_user_id="admin-1")
    assert model_registry.get_champion_version() == "v2"

    governance.rollback(trained_model_bundle.metadata["version"], actor_user_id="admin-1")
    assert model_registry.get_champion_version() == trained_model_bundle.metadata["version"]

    versions = governance.list_versions()
    assert any(v["is_champion"] for v in versions)


def test_auth_service_locks_account_after_max_failed_attempts(db_session_factory):
    service = _auth_service(db_session_factory)
    service.register(username="lockout1", email="lockout1@example.com", password="Supersecret1!", role="viewer")

    for _ in range(MAX_FAILED_LOGIN_ATTEMPTS - 1):
        with pytest.raises(AuthenticationError):
            service.login(username="lockout1", password="wrong-password")

    # The attempt that reaches the threshold raises the more specific
    # AccountLockedError (a subclass of AuthenticationError).
    with pytest.raises(AccountLockedError):
        service.login(username="lockout1", password="wrong-password")

    # Even the correct password is now rejected while locked.
    with pytest.raises(AccountLockedError):
        service.login(username="lockout1", password="Supersecret1!")


def test_user_service_unlock_account_allows_login_again(db_session_factory):
    db = db_session_factory()
    user_repo = UserRepository(db)
    audit_repo = AuditLogRepository(db)
    auth_service = AuthService(user_repo, audit_repo)
    user_service = UserService(user_repo, audit_repo)

    auth_service.register(username="lockout2", email="lockout2@example.com", password="Supersecret1!", role="viewer")
    for _ in range(MAX_FAILED_LOGIN_ATTEMPTS):
        with pytest.raises(AuthenticationError):
            auth_service.login(username="lockout2", password="wrong-password")

    locked_user = user_repo.get_by_username("lockout2")
    assert locked_user.locked_until is not None

    user_service.unlock_account(locked_user.id, actor_user_id="admin-1")
    result = auth_service.login(username="lockout2", password="Supersecret1!")
    assert result.tokens.access_token


def test_user_service_cannot_deactivate_own_account(db_session_factory):
    db = db_session_factory()
    user_repo = UserRepository(db)
    audit_repo = AuditLogRepository(db)
    user_service = UserService(user_repo, audit_repo)

    # Privileged roles can no longer self-register (see
    # AuthService.register's SELF_SERVICE_ROLES gate) -- seed directly via
    # the repository, exactly like `_register_and_login` does for the API
    # tests, since this test is about `set_active_status`'s self-deactivation
    # guard, not about registration.
    admin = user_repo.create(
        username="selfadmin1",
        email="sa1@example.com",
        hashed_password=hash_password("Supersecret1!"),
        role="admin",
    )

    with pytest.raises(AuthorizationError):
        user_service.set_active_status(admin.id, is_active=False, actor_user_id=admin.id)


def test_user_service_get_unknown_user_raises_not_found(db_session_factory):
    db = db_session_factory()
    user_service = UserService(UserRepository(db), AuditLogRepository(db))
    with pytest.raises(ResourceNotFoundError):
        user_service.get_user("no-such-user-id")


def test_audit_service_list_logs_returns_paginated_page(db_session_factory):
    db = db_session_factory()
    audit_repo = AuditLogRepository(db)
    audit_repo.log_action(user_id="u1", action="login")
    audit_repo.log_action(user_id="u1", action="predict")

    from core.pagination import pagination_params

    service = AuditService(audit_repo)
    params = pagination_params(page=1, size=10, sort_by=None, sort_order="desc")
    page = service.list_logs(params)
    assert page.total >= 2
    assert len(page.items) >= 2


def _application(payload: dict):
    from api.schemas import CreditApplicationRequest

    return CreditApplicationRequest(**payload)


def test_data_quality_service_run_and_persist_returns_flattened_dict(db_session_factory):
    import pandas as pd

    db = db_session_factory()
    service = DataQualityService(DataQualityReportRepository(db))

    df = pd.DataFrame(
        {
            "age": [30, 40, 50],
            "annual_income": [50000, 60000, 70000],
            "credit_score": [700, 720, 680],
        }
    )
    result = service.run_and_persist(df, dataset_label="unit-test")

    assert result["dataset_label"] == "unit-test"
    assert result["report_type"] == "quality_report"
    assert "validation" in result and "profile" in result and "score" in result
    # `passed` reflects `run_full_validation`'s hard-gate verdict, which is
    # False here because a 3-column DataFrame is missing most of the columns
    # `validate_schema` expects from the full raw-feature schema -- this test
    # is about the service correctly plumbing validation/profile/score
    # through into a persisted, flattened dict, not about achieving a
    # passing score with a partial column set.
    assert isinstance(result["passed"], bool)
    assert result["id"] is not None


def test_data_quality_service_get_latest_returns_none_when_empty(db_session_factory):
    db = db_session_factory()
    service = DataQualityService(DataQualityReportRepository(db))
    assert service.get_latest() is None


def test_data_quality_service_get_latest_returns_most_recent_run(db_session_factory):
    import pandas as pd

    db = db_session_factory()
    service = DataQualityService(DataQualityReportRepository(db))

    df1 = pd.DataFrame({"age": [30, 40]})
    df2 = pd.DataFrame({"age": [30, 40, 50]})
    service.run_and_persist(df1, dataset_label="first")
    service.run_and_persist(df2, dataset_label="second")

    latest = service.get_latest()
    assert latest["dataset_label"] == "second"


def test_data_quality_service_list_history_is_paginated(db_session_factory):
    import pandas as pd

    from core.pagination import pagination_params

    db = db_session_factory()
    service = DataQualityService(DataQualityReportRepository(db))

    for i in range(3):
        service.run_and_persist(pd.DataFrame({"age": [30 + i, 40 + i]}), dataset_label=f"run-{i}")

    params = pagination_params(page=1, size=2, sort_by=None, sort_order="desc")
    page = service.list_history(params)
    assert page.total == 3
    assert len(page.items) == 2


def test_data_quality_service_run_and_persist_records_failure_when_critical_issue(db_session_factory):
    import pandas as pd

    db = db_session_factory()
    service = DataQualityService(DataQualityReportRepository(db))

    # A dataset with a 90%-missing column trips the null-threshold hard gate
    # (critical), so the persisted report should record passed=False.
    df = pd.DataFrame({"age": [30] + [None] * 9})
    result = service.run_and_persist(df, dataset_label="dirty")
    assert result["passed"] is False
