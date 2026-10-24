"""
MLflow experiment tracking + model registry wrapper.

Design decision: rather than sprinkling raw `mlflow.log_*` calls throughout
the training pipeline, this module centralizes MLflow interaction behind a
small `ExperimentTracker` class. That gives us one place to (a) configure the
tracking URI consistently, (b) enforce a consistent naming/tagging scheme
across runs so the comparison dashboard can query reliably, and (c) swap the
backend (local file store -> remote tracking server) without touching
training code.

Reliability (this pass): a bare local filesystem path (this project's own
default, `str(PROJECT_ROOT / "mlruns")`) is not a valid URI. On Windows this
actively breaks MLflow: `F:\\credit-risk-platform\\mlruns` gets misparsed by
`urllib.parse.urlparse` as scheme `f` (a drive letter looks exactly like a
URI scheme to a naive parser), which fails MLflow's internal tracking/
registry store resolution outright. `_normalize_tracking_uri()` converts a
bare local path to a proper `file://` URI up front, on every platform --
this was previously only ever exercised with `http://` tracking servers or
POSIX paths (which don't hit this specific ambiguity), so no test caught it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from core.logging import get_logger

logger = get_logger(__name__)


def _normalize_tracking_uri(tracking_uri: str) -> str:
    if "://" in tracking_uri or tracking_uri.startswith("databricks"):
        return tracking_uri
    # A bare local filesystem path, not a URI -- ensure the directory exists,
    # then convert to a `file://` URI so MLflow's own URI parsing never
    # misreads a Windows drive letter as a URI scheme.
    path = Path(tracking_uri).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


class ExperimentTracker:
    def __init__(self, tracking_uri: str, experiment_name: str):
        tracking_uri = _normalize_tracking_uri(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.experiment_name = experiment_name

    def start_run(self, run_name: str, tags: dict[str, str] | None = None):
        return mlflow.start_run(run_name=run_name, tags=tags or {})

    @staticmethod
    def log_params(params: dict[str, Any]) -> None:
        # MLflow rejects values >250 chars / non-primitive types silently mangling them;
        # stringify defensively so nested dicts (e.g. search spaces) don't break logging.
        safe = {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v) for k, v in params.items()}
        mlflow.log_params(safe)

    @staticmethod
    def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    @staticmethod
    def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    @staticmethod
    def log_figure(figure, artifact_file: str) -> None:
        mlflow.log_figure(figure, artifact_file)

    @staticmethod
    def log_sklearn_model(model, artifact_path: str, registered_model_name: str | None = None) -> None:
        mlflow.sklearn.log_model(model, name=artifact_path, registered_model_name=registered_model_name)

    def get_best_run(self, metric: str = "pr_auc", ascending: bool = False) -> dict | None:
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return None
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )
        if not runs:
            return None
        run = runs[0]
        return {"run_id": run.info.run_id, "metrics": run.data.metrics, "params": run.data.params}

    def promote_to_production(self, model_name: str, version: str) -> None:
        """Transition a registered model version to Production, archiving the previous one.

        This mirrors what a real registry-gated deployment looks like: the
        API service loads whichever version currently holds the
        'Production' alias/stage rather than a hardcoded path, so promotion
        is a metadata operation, not a redeploy.
        """
        current_prod = self.client.get_latest_versions(model_name, stages=["Production"])
        for mv in current_prod:
            self.client.transition_model_version_stage(model_name, mv.version, stage="Archived")
        self.client.transition_model_version_stage(model_name, version, stage="Production")
        logger.info(
            "Model promoted to Production",
            extra={"context": {"model_name": model_name, "version": version}},
        )

    def rollback(self, model_name: str, target_version: str) -> None:
        """Rolls back Production to a specific earlier version — the whole point
        of a model registry versus overwriting a single model file on disk."""
        self.promote_to_production(model_name, target_version)
        logger.warning(
            "Model ROLLED BACK",
            extra={"context": {"model_name": model_name, "rolled_back_to_version": target_version}},
        )
