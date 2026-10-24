"""Unit tests for `mlflow_tracking/tracker.py`'s tracking-URI normalization.

Regression coverage for a real bug: a bare local filesystem path (this
project's own default `mlflow_tracking_uri`, `str(PROJECT_ROOT / "mlruns")`)
is not a valid URI. On Windows, `F:\\credit-risk-platform\\mlruns` was
misparsed by `urllib.parse.urlparse` as scheme `f` (a drive letter looks
exactly like a URI scheme), which broke MLflow's internal store resolution
outright -- `ExperimentTracker.__init__()` raised before a single param/
metric could be logged. No test exercised this before because it only
manifests with a genuine local Windows path, not the `http://`/POSIX paths
prior testing happened to use.
"""

from __future__ import annotations

import os

import pytest

from mlflow_tracking.tracker import ExperimentTracker, _normalize_tracking_uri


@pytest.mark.parametrize(
    "uri",
    [
        "http://mlflow-server:5000",
        "https://mlflow-server:5000",
        "sqlite:///mlflow.db",
        "postgresql://user:pass@host/db",
        "databricks",
    ],
)
def test_normalize_tracking_uri_passes_through_real_uris_unchanged(uri):
    assert _normalize_tracking_uri(uri) == uri


def test_normalize_tracking_uri_converts_bare_local_path_to_file_uri(tmp_path):
    bare_path = str(tmp_path / "mlruns")
    normalized = _normalize_tracking_uri(bare_path)
    assert normalized.startswith("file:")
    assert os.path.isdir(bare_path)  # directory created as a side effect


def test_normalize_tracking_uri_windows_style_path_never_looks_like_a_scheme(tmp_path):
    """The exact shape of the original bug: a path with a drive-letter-like
    prefix must not round-trip as something `urllib.parse.urlparse` would
    misread as a URI scheme."""
    from urllib.parse import urlparse

    windows_style_path = str(tmp_path / "mlruns")
    normalized = _normalize_tracking_uri(windows_style_path)
    assert urlparse(normalized).scheme in ("file", "")


def test_experiment_tracker_construction_and_run_works_with_bare_local_path(tmp_path, monkeypatch):
    """End-to-end regression test: constructing `ExperimentTracker` with a
    bare local path and starting a run must not raise -- this is exactly
    the call path `pipelines/run_training.py`'s `_tracker()` helper uses
    with `settings.mlflow_tracking_uri`."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracker = ExperimentTracker(str(tmp_path / "mlruns"), "test-experiment")
    with tracker.start_run(run_name="test_run"):
        tracker.log_params({"a": 1})
        tracker.log_metrics({"roc_auc": 0.9})
