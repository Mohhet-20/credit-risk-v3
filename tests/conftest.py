"""
Shared pytest fixtures.

Design decision: tests use a SMALL synthetic dataset (500 rows) and a
single fast LogisticRegression model rather than the full Optuna-tuned
5-model comparison -- the training pipeline's correctness is what's under
test here, not model quality, so keeping fixtures fast (sub-second) is more
valuable than test-time realism. API tests use FastAPI's `dependency_overrides`
(the framework-recommended pattern) to inject an in-memory SQLite session and
a fixture-built model bundle, rather than fighting the process-wide
`get_settings()`/model-cache singletons.

Design decision: the in-memory SQLite engine uses `StaticPool` so every
connection (including the one FastAPI's request-handling thread opens)
shares the same underlying `:memory:` database instead of each getting its
own empty one.
"""

from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import core.sklearn_compat  # noqa: F401
from core.constants import TARGET_COLUMN
from data.generate_synthetic_data import generate_credit_dataset
from database.models import Base
from feature_store.pipeline import FeaturePipeline
from models.registry import ModelBundle, ModelRegistry
from training.calibration import calibrate_model


@pytest.fixture(scope="session")
def raw_dataset() -> pd.DataFrame:
    return generate_credit_dataset(n_samples=500, seed=7)


@pytest.fixture(scope="session")
def raw_dataset_no_leak(raw_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = raw_dataset[TARGET_COLUMN]
    X = raw_dataset.drop(columns=[TARGET_COLUMN])
    return X, y


@pytest.fixture(scope="session")
def fitted_feature_pipeline(raw_dataset_no_leak) -> FeaturePipeline:
    X, y = raw_dataset_no_leak
    fp = FeaturePipeline(importance_top_k=None, random_state=7)
    fp.fit(X, y)
    return fp


@pytest.fixture(scope="session")
def transformed_data(fitted_feature_pipeline, raw_dataset_no_leak):
    X, y = raw_dataset_no_leak
    return fitted_feature_pipeline.transform(X), y.reset_index(drop=True)


@pytest.fixture(scope="session")
def trained_model_bundle(fitted_feature_pipeline, transformed_data, tmp_path_factory) -> ModelBundle:
    Xt, y = transformed_data
    split = int(len(Xt) * 0.7)
    X_train, y_train = Xt.iloc[:split], y.iloc[:split]
    X_calib, y_calib = Xt.iloc[split:], y.iloc[split:]

    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    calibrated = calibrate_model(model, X_calib, y_calib, method="sigmoid", cv=3)

    metadata = {
        "model_name": "logistic_regression",
        "version": "test-v1",
        "decision_threshold": 0.2,
        "registered_at": "2024-01-01T00:00:00",
    }
    return ModelBundle(model=calibrated, feature_pipeline=fitted_feature_pipeline, metadata=metadata)


@pytest.fixture()
def model_registry(tmp_path, trained_model_bundle) -> ModelRegistry:
    registry = ModelRegistry(tmp_path / "artifacts")
    registry.register_version(
        version=trained_model_bundle.metadata["version"],
        model=trained_model_bundle.model,
        feature_pipeline=trained_model_bundle.feature_pipeline,
        metadata=trained_model_bundle.metadata,
    )
    registry.set_champion(trained_model_bundle.metadata["version"])
    return registry


@pytest.fixture()
def db_session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


@pytest.fixture()
def api_client(db_session_factory, trained_model_bundle):
    from fastapi.testclient import TestClient

    from api.app import app
    from api.dependencies import get_model_bundle, get_model_registry
    from core.rate_limit import limiter
    from database.session import get_db

    # Rate limiting (global default + the stricter per-route limits on
    # /auth/login and /predict/batch, see core/rate_limit.py) is disabled
    # for the general test suite: most tests register/login many times in
    # quick succession against the same in-process limiter state, which has
    # nothing to do with what each individual test is actually verifying.
    # `tests/api/test_rate_limiting.py` re-enables it explicitly to test the
    # limiter itself.
    limiter.enabled = False

    def _override_get_db():
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()

    def _override_get_model_bundle():
        return trained_model_bundle

    class _StubRegistry:
        def get_champion_version(self_inner):
            return trained_model_bundle.metadata["version"]

        def load_version(self_inner, version):
            return trained_model_bundle

        def list_versions(self_inner):
            return [trained_model_bundle.metadata]

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_model_bundle] = _override_get_model_bundle
    app.dependency_overrides[get_model_registry] = lambda: _StubRegistry()

    with TestClient(app) as client:
        # Exposed so test helpers (e.g. `tests/api/test_auth_and_predict.py::_register_and_login`)
        # can open a DB session against the SAME in-memory SQLite engine this
        # client is wired to, for arrange-phase setup that bypasses the HTTP
        # layer entirely (e.g. seeding a privileged-role user directly via the
        # repository, now that POST /auth/register is restricted to self-service
        # roles -- see AuthService.register's SELF_SERVICE_ROLES gate).
        client.db_session_factory = db_session_factory
        yield client

    app.dependency_overrides.clear()
    limiter.enabled = True
