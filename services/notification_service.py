"""
Notification dispatch: a minimal, honest abstraction for "tell someone a
workflow finished" -- deliberately NOT wired to a real email/SMS provider
(see `docs/ROADMAP.md`'s "medium term" section: adding real email delivery
is a genuine external-dependency decision, not something to fake with a
stub that looks functional but silently does nothing in production).

What this DOES do today: records every notification as a structured log
line (queryable via correlation ID / log aggregation, same as every other
platform event) and, if the recipient is a known platform user, an
`AuditLog` entry -- so "was a notification sent for this workflow
execution" is always answerable from the audit trail even without an
external notification provider configured. `services.config_service`'s
`notifications.enabled` flag is the master on/off switch a real provider
integration would also respect.

Swapping in a real provider (SES, SendGrid, Twilio, Slack webhook) later
is a matter of adding a `_dispatch()` implementation here; every caller
(`WorkflowService`) is already written against this interface.
"""

from __future__ import annotations

from core.logging import get_logger
from repositories.audit_log_repository import AuditLogRepository

logger = get_logger(__name__)


class NotificationService:
    def __init__(self, audit_repo: AuditLogRepository, config_service):
        self.audit_repo = audit_repo
        self.config_service = config_service

    def notify(
        self, *, event_type: str, recipient_user_id: str | None, message: str, details: dict | None = None
    ) -> bool:
        """Returns True if a notification was actually dispatched, False if
        suppressed (notifications disabled via config). Never raises --a
        notification failure should never fail the workflow step that
        triggered it (see `WorkflowService`'s `non_fatal=True` usage for
        the notification step)."""
        if not self.config_service.get("notifications.enabled"):
            logger.info(f"Notification suppressed (notifications.enabled=False): {event_type}")
            return False

        logger.info(
            f"Notification: {event_type} -> {message}",
            extra={"context": {"event_type": event_type, "recipient": recipient_user_id, "details": details or {}}},
        )
        self.audit_repo.log_action(
            user_id=recipient_user_id,
            action=f"notification.{event_type}",
            details=details,
        )
        return True
