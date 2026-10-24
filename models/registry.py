"""
Local model artifact registry.

Design decision: the MLflow Model Registry is the system of record for
experiment history and stage transitions, but the FastAPI service needs a
FAST, dependency-free way to load "whatever is currently Production" without
requiring a live MLflow tracking server to be reachable at request time
(unacceptable latency/availability coupling for a serving path). This module
is a thin local mirror: every promoted model is copied into
`models/artifacts/<version>/` alongside its FeaturePipeline and metadata,
and a `champion.json` pointer file records the active version. The API reads
only this pointer file, keeping serving fully decoupled from MLflow uptime.
Promotion (`set_champion`) and rollback (`rollback_to`) are simple pointer
rewrites, mirroring how the MLflow stage transition works but without the
network dependency.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from core.exceptions import ModelLoadError, ModelNotFoundError
from core.logging import get_logger

logger = get_logger(__name__)

CHAMPION_POINTER_FILENAME = "champion.json"


def _atomic_write_text(path: Path, content: str) -> None:
    """Writes `content` to `path` via a temp file + `os.replace()` instead of
    a direct `write_text()`. `os.replace()` is atomic on both POSIX and
    Windows within the same filesystem, so a concurrent reader (e.g. every
    `/predict` request calling `load_champion()` while `set_champion()`
    promotes a new version) always sees either the old, complete content or
    the new, complete content -- never a partially-written file."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


@dataclass
class ModelBundle:
    model: Any
    feature_pipeline: Any
    metadata: dict


class ModelRegistry:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def register_version(
        self,
        version: str,
        model: Any,
        feature_pipeline: Any,
        metadata: dict,
    ) -> Path:
        version_dir = self.artifact_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, version_dir / "model.joblib")
        joblib.dump(feature_pipeline, version_dir / "feature_pipeline.joblib")

        metadata = {**metadata, "version": version, "registered_at": datetime.now(timezone.utc).isoformat()}
        _atomic_write_text(version_dir / "metadata.json", json.dumps(metadata, indent=2, default=str))

        logger.info("Registered model version", extra={"context": {"version": version}})
        return version_dir

    def set_champion(self, version: str) -> None:
        version_dir = self.artifact_dir / version
        if not version_dir.exists():
            raise ModelNotFoundError(f"Cannot promote unknown version: {version}")

        pointer = {"champion_version": version, "promoted_at": datetime.now(timezone.utc).isoformat()}
        _atomic_write_text(self.artifact_dir / CHAMPION_POINTER_FILENAME, json.dumps(pointer, indent=2))
        logger.info("Champion updated", extra={"context": {"version": version}})

    def rollback_to(self, version: str) -> None:
        logger.warning("Rolling back champion model", extra={"context": {"target_version": version}})
        self.set_champion(version)

    def get_champion_version(self) -> str | None:
        pointer_path = self.artifact_dir / CHAMPION_POINTER_FILENAME
        if not pointer_path.exists():
            return None
        return json.loads(pointer_path.read_text())["champion_version"]

    def load_champion(self) -> ModelBundle:
        version = self.get_champion_version()
        if version is None:
            raise ModelNotFoundError("No champion model has been promoted yet.")
        return self.load_version(version)

    def load_version(self, version: str) -> ModelBundle:
        version_dir = self.artifact_dir / version
        if not version_dir.exists():
            raise ModelNotFoundError(f"Model version not found: {version}")
        try:
            model = joblib.load(version_dir / "model.joblib")
            feature_pipeline = joblib.load(version_dir / "feature_pipeline.joblib")
            metadata = json.loads((version_dir / "metadata.json").read_text())
        except Exception as exc:  # noqa: BLE001 - surface as a typed platform error
            raise ModelLoadError(f"Failed to load model version {version}: {exc}") from exc
        return ModelBundle(model=model, feature_pipeline=feature_pipeline, metadata=metadata)

    def list_versions(self) -> list[dict]:
        versions = []
        for d in sorted(self.artifact_dir.iterdir()):
            if d.is_dir() and (d / "metadata.json").exists():
                versions.append(json.loads((d / "metadata.json").read_text()))
        return sorted(versions, key=lambda m: m.get("registered_at", ""), reverse=True)

    def delete_version(self, version: str) -> None:
        version_dir = self.artifact_dir / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
            logger.info("Deleted model version", extra={"context": {"version": version}})
