"""
Generic pagination envelope + FastAPI query-param dependency.

Design decision: page/size (not offset/limit) is the public API shape --
`page` is 1-indexed and more intuitive for API consumers than a raw offset,
while repositories internally translate to `limit`/`offset` for the SQL
query (`PaginationParams.to_offset_limit()`). `sort_by`/`sort_order` are
plain strings that each repository must validate against its own explicit
allowlist of sortable columns -- never interpolate a caller-supplied string
directly into a query -- so a caller can never sort by an arbitrary,
unindexed, or injected column.
"""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from fastapi import Query
from pydantic import BaseModel

T = TypeVar("T")

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 200


class PaginationParams(BaseModel):
    page: int = 1
    size: int = DEFAULT_PAGE_SIZE
    sort_by: str | None = None
    sort_order: Literal["asc", "desc"] = "desc"

    def to_offset_limit(self) -> tuple[int, int]:
        return (self.page - 1) * self.size, self.size


def pagination_params(
    page: int = Query(1, ge=1, description="1-indexed page number"),
    size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Items per page"),
    sort_by: str | None = Query(None, description="Column to sort by"),
    sort_order: Literal["asc", "desc"] = Query("desc"),
) -> PaginationParams:
    return PaginationParams(page=page, size=size, sort_by=sort_by, sort_order=sort_order)


class Page(BaseModel, Generic[T]):
    items: list[T]
    page: int
    size: int
    total: int
    total_pages: int

    @classmethod
    def create(cls, items: list[T], total: int, params: PaginationParams) -> Page[T]:
        total_pages = (total + params.size - 1) // params.size if params.size else 0
        return cls(items=items, page=params.page, size=params.size, total=total, total_pages=total_pages)
