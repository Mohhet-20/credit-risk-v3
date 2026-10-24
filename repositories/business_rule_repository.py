"""Repository for `BusinessRule` rows -- see `rules/engine.py` for
evaluation logic and `services/rule_service.py` for the business-logic
layer (versioning, audit-log snapshots) built on top of this."""

from __future__ import annotations

from sqlalchemy import select

from database.models import BusinessRule
from repositories.base import BaseRepository


class BusinessRuleRepository(BaseRepository[BusinessRule]):
    model = BusinessRule

    def list_active(self) -> list[BusinessRule]:
        stmt = select(BusinessRule).where(BusinessRule.is_active.is_(True)).order_by(BusinessRule.priority.asc())
        return list(self.db.scalars(stmt))

    def list_all_rules(self, *, rule_type: str | None = None) -> list[BusinessRule]:
        stmt = select(BusinessRule).order_by(BusinessRule.priority.asc())
        if rule_type:
            stmt = stmt.where(BusinessRule.rule_type == rule_type)
        return list(self.db.scalars(stmt))

    def update(self, rule: BusinessRule, **fields) -> BusinessRule:
        for key, value in fields.items():
            setattr(rule, key, value)
        rule.version += 1
        self.db.commit()
        self.db.refresh(rule)
        return rule
