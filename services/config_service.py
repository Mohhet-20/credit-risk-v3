"""
Enterprise Configuration Center service layer: get/set/list admin-editable
system configuration (feature flags, thresholds, notification settings,
workflow toggles, model-selection preferences) backed by `SystemConfig`
(see its docstring in `database/models.py` for the single-table design
decision). Every write is audit-logged, same pattern as `RuleService`.

Design decision: `DEFAULT_CONFIG` seeds a small set of well-known keys with
safe defaults (documented inline) so the platform behaves sensibly even
before an admin has touched the Configuration Center at all -- `get()`
falls back to a default rather than returning `None`/raising for any key
listed here, so callers (the workflow engine checking whether the fraud-
check step is enabled, for instance) never need a None-check for a config
key that simply hasn't been explicitly set yet.
"""

from __future__ import annotations

from repositories.audit_log_repository import AuditLogRepository
from repositories.system_config_repository import SystemConfigRepository

DEFAULT_CONFIG: dict[str, dict] = {
    "workflow.fraud_check_enabled": {
        "value": False,
        "category": "workflow",
        "description": "Whether the workflow engine's fraud-check step runs (placeholder architecture -- see plugins/fraud_detection/).",
    },
    "workflow.manual_review_enabled": {
        "value": True,
        "category": "workflow",
        "description": "Whether borderline predictions are routed to a manual-review step.",
    },
    "workflow.manual_review_band": {
        "value": {"lower": 0.45, "upper": 0.55},
        "category": "workflow",
        "description": "Predicted-probability band that triggers manual review when no business rule already decided the outcome.",
    },
    "notifications.enabled": {
        "value": True,
        "category": "notifications",
        "description": "Master switch for workflow-completion notifications (see services/notification_service.py).",
    },
    "feature_flags.copilot_enabled": {
        "value": True,
        "category": "feature_flags",
        "description": "Master switch for the /copilot endpoints.",
    },
    "feature_flags.rule_engine_enabled": {
        "value": True,
        "category": "feature_flags",
        "description": "Master switch for business-rule evaluation during scoring/workflow execution.",
    },
}


class ConfigService:
    def __init__(self, config_repo: SystemConfigRepository, audit_repo: AuditLogRepository):
        self.config_repo = config_repo
        self.audit_repo = audit_repo

    def get(self, key: str):
        """Returns the configured value for `key`, falling back to
        `DEFAULT_CONFIG[key]["value"]` if no admin override has been
        persisted yet, or `None` if `key` isn't a recognized default
        either (an unrecognized key is not an error -- callers may use
        forward-compatible keys not yet in `DEFAULT_CONFIG`)."""
        record = self.config_repo.get_by_key(key)
        if record is not None:
            return record.value
        default = DEFAULT_CONFIG.get(key)
        return default["value"] if default else None

    def list_all(self, *, category: str | None = None) -> list[dict]:
        """Returns every config key currently known to the platform --
        persisted overrides AND still-at-default keys -- each tagged with
        whether it's been explicitly overridden, so the Configuration
        Center UI can show "using default" vs. "admin-configured"."""
        persisted = {r.key: r for r in self.config_repo.list_by_category(category)}
        results = []
        seen = set()
        for key, record in persisted.items():
            seen.add(key)
            results.append(
                {
                    "key": key,
                    "value": record.value,
                    "category": record.category,
                    "description": record.description,
                    "is_default": False,
                    "updated_at": record.updated_at.isoformat() if record.updated_at else None,
                }
            )
        for key, meta in DEFAULT_CONFIG.items():
            if key in seen:
                continue
            if category and meta["category"] != category:
                continue
            results.append(
                {
                    "key": key,
                    "value": meta["value"],
                    "category": meta["category"],
                    "description": meta["description"],
                    "is_default": True,
                    "updated_at": None,
                }
            )
        return sorted(results, key=lambda r: r["key"])

    def set(self, key: str, value, *, category: str, description: str | None, actor_user_id: str | None) -> dict:
        record = self.config_repo.upsert(
            key=key,
            value=value,
            category=category,
            description=description,
            updated_by=actor_user_id,
        )
        self.audit_repo.log_action(
            user_id=actor_user_id,
            action="config.update",
            resource=f"config:{key}",
            details={"category": category, "value": value},
        )
        return {
            "key": record.key,
            "value": record.value,
            "category": record.category,
            "description": record.description,
            "is_default": False,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        }
