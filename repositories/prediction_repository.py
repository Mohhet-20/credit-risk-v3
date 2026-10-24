"""Prediction aggregate repository."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from database.models import Prediction
from repositories.base import BaseRepository


class PredictionRepository(BaseRepository[Prediction]):
    model = Prediction

    def create(self, **fields) -> Prediction:
        return self.add(Prediction(**fields))

    def list_predictions(
        self,
        limit: int = 100,
        offset: int = 0,
        customer_id: str | None = None,
    ) -> list[Prediction]:
        stmt = select(Prediction)
        if customer_id:
            stmt = stmt.where(Prediction.customer_id == customer_id)
        stmt = stmt.order_by(Prediction.created_at.desc()).limit(limit).offset(offset)
        return list(self.db.scalars(stmt))

    def list_since(self, since: datetime | None = None, limit: int = 2000) -> list[Prediction]:
        stmt = select(Prediction)
        if since:
            stmt = stmt.where(Prediction.created_at >= since)
        stmt = stmt.order_by(Prediction.created_at.desc()).limit(limit)
        return list(self.db.scalars(stmt))

    def stats_last_n_days(self, days: int = 30) -> dict:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        stmt = select(
            func.count(Prediction.id),
            func.avg(Prediction.predicted_probability),
            func.avg(Prediction.latency_ms),
        ).where(Prediction.created_at >= since)
        count, avg_proba, avg_latency = self.db.execute(stmt).one()
        return {
            "n_predictions": int(count or 0),
            "avg_predicted_probability": float(avg_proba) if avg_proba is not None else None,
            "avg_latency_ms": float(avg_latency) if avg_latency is not None else None,
            "window_days": days,
        }
