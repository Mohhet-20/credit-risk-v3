"""Model version record aggregate repository.

Mirrors the local `ModelRegistry` / MLflow registry into the relational
database for fast dashboard queries -- the registry filesystem/MLflow store
remains the source of truth for the artifacts themselves (see
`models/registry.py`); this table is a queryable read model of it.
"""

from __future__ import annotations

from sqlalchemy import select

from database.models import ModelVersionRecord
from repositories.base import BaseRepository


class ModelVersionRepository(BaseRepository[ModelVersionRecord]):
    model = ModelVersionRecord

    def upsert(
        self,
        *,
        version: str,
        model_name: str,
        stage: str = "None",
        metrics: dict | None = None,
        decision_threshold: float | None = None,
    ) -> ModelVersionRecord:
        existing = self.db.scalar(select(ModelVersionRecord).where(ModelVersionRecord.version == version))
        if existing:
            existing.stage = stage
            existing.metrics = metrics
            existing.decision_threshold = decision_threshold
            self.db.commit()
            self.db.refresh(existing)
            return existing
        record = ModelVersionRecord(
            version=version,
            model_name=model_name,
            stage=stage,
            metrics=metrics,
            decision_threshold=decision_threshold,
        )
        return self.add(record)

    def list_model_versions(self, limit: int = 100, offset: int = 0) -> list[ModelVersionRecord]:
        stmt = select(ModelVersionRecord).order_by(ModelVersionRecord.registered_at.desc()).limit(limit).offset(offset)
        return list(self.db.scalars(stmt))
