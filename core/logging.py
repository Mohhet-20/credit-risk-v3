"""
Structured logging configuration shared by every subsystem (training,
API, monitoring, batch jobs).

Design decision: we use stdlib `logging` with a JSON formatter rather than
pulling in structlog/loguru. In an enterprise setting the platform team
usually already standardizes on stdlib logging + a log shipper (Fluent Bit,
Vector, CloudWatch agent) that expects JSON lines on stdout — adding another
logging framework increases operational surface area for no real benefit.

Design decision: request correlation IDs are threaded through as a
`contextvars.ContextVar` + a `logging.Filter` attached to the root handler,
rather than requiring every call site to pass `extra={"context": {...}}`
manually. `api.middleware.CorrelationIdMiddleware` calls `set_correlation_id`
once per request; every log line emitted anywhere in that request's call
stack (router, service, repository, deep inside a background task) is
tagged automatically by `CorrelationIdFilter` before it reaches the
formatter. This keeps correlation-ID propagation entirely out of business
logic — no service or repository method needs to know it exists.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

_correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def set_correlation_id(correlation_id: str | None) -> None:
    _correlation_id_var.set(correlation_id)


def get_correlation_id() -> str | None:
    return _correlation_id_var.get()


class CorrelationIdFilter(logging.Filter):
    """Injects the active request's correlation ID (if any) into every log
    record's `context` dict — the same dict `JSONFormatter`/`HumanFormatter`
    already read from `extra={"context": {...}}`, so no formatter changes
    are needed to make it visible."""

    def filter(self, record: logging.LogRecord) -> bool:
        correlation_id = get_correlation_id()
        if correlation_id:
            context = getattr(record, "context", None)
            if context is None:
                record.context = {"correlation_id": correlation_id}
            else:
                context.setdefault("correlation_id", correlation_id)
        return True


class JSONFormatter(logging.Formatter):
    """Renders log records as single-line JSON for machine ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Allow callers to attach structured context via `extra={"context": {...}}`
        context = getattr(record, "context", None)
        if context:
            payload["context"] = context
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Readable console formatter used in local development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = getattr(record, "context", None)
        if context:
            base = f"{base} | {json.dumps(context, default=str)}"
        return base


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Idempotently configure the root logger. Safe to call multiple times."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if json_logs else HumanFormatter())
    handler.addFilter(CorrelationIdFilter())
    root.addHandler(handler)

    # Quiet down noisy third-party loggers.
    for noisy in ("uvicorn.access", "urllib3", "matplotlib", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
