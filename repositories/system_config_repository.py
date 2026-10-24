"""Repository for `SystemConfig` rows -- the Enterprise Configuration
Center's storage layer. See `services/config_service.py`."""

from __future__ import annotations

from sqlalchemy import select

from database.models import SystemConfig
from repositories.base import BaseRepository


class SystemConfigRepository(BaseRepository[SystemConfig]):
    model = SystemConfig

    def get_by_key(self, key: str) -> SystemConfig | None:
        return self.db.scalar(select(SystemConfig).where(SystemConfig.key == key))

    def list_by_category(self, category: str | None = None) -> list[SystemConfig]:
        stmt = select(SystemConfig).order_by(SystemConfig.category, SystemConfig.key)
        if category:
            stmt = stmt.where(SystemConfig.category == category)
        return list(self.db.scalars(stmt))

    def upsert(self, *, key: str, value: dict, category: str, description: str | None, updated_by: str | None):
        existing = self.get_by_key(key)
        if existing:
            existing.value = value
            existing.category = category
            existing.description = description
            existing.updated_by = updated_by
            self.db.commit()
            self.db.refresh(existing)
            return existing
        record = SystemConfig(
            key=key,
            value=value,
            category=category,
            description=description,
            updated_by=updated_by,
        )
        return self.add(record)
