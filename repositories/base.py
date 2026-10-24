"""
Generic base repository.

Design decision: a small `BaseRepository[ModelT]` covers the truly identical
CRUD operations (get by ID, list with limit/offset, add, delete) that every
concrete repository would otherwise reimplement. Aggregate-specific queries
(e.g. "predictions in the last N days", "user by username") stay on the
concrete subclasses rather than being forced into the generic base — a
generic repository that tries to express every possible query becomes an
unreadable leaky abstraction, which is exactly the KISS/YAGNI tradeoff this
codebase prefers (see `database/crud.py`'s original docstring on the same
tradeoff, before this module superseded it).
"""

from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    model: type[ModelT]

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, entity_id: str) -> ModelT | None:
        return self.db.get(self.model, entity_id)

    def list_all(self, limit: int = 100, offset: int = 0) -> list[ModelT]:
        stmt = select(self.model).limit(limit).offset(offset)
        return list(self.db.scalars(stmt))

    def add(self, entity: ModelT) -> ModelT:
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    def delete(self, entity: ModelT) -> None:
        self.db.delete(entity)
        self.db.commit()
