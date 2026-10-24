"""
Business Rule Engine service layer: CRUD for admin-editable rules plus
evaluation/simulation, orchestrating `rules/engine.py` (pure logic) with
`BusinessRuleRepository` (persistence) and `AuditLogRepository` (change
history -- see `BusinessRule`'s docstring in `database/models.py` for why
rule history reuses the audit log rather than a dedicated table).
"""

from __future__ import annotations

from core.exceptions import ResourceNotFoundError
from database.models import BusinessRule
from repositories.audit_log_repository import AuditLogRepository
from repositories.business_rule_repository import BusinessRuleRepository
from rules.engine import VALID_RULE_TYPES, evaluate_rules, simulate_rules, validate_condition


class InvalidRuleTypeError(ValueError):
    pass


class RuleService:
    def __init__(self, rule_repo: BusinessRuleRepository, audit_repo: AuditLogRepository):
        self.rule_repo = rule_repo
        self.audit_repo = audit_repo

    def create_rule(
        self,
        *,
        name: str,
        description: str | None,
        rule_type: str,
        condition: dict,
        priority: int,
        actor_user_id: str | None,
    ) -> BusinessRule:
        if rule_type not in VALID_RULE_TYPES:
            raise InvalidRuleTypeError(f"rule_type must be one of {VALID_RULE_TYPES}, got '{rule_type}'.")
        validate_condition(condition)  # raises InvalidRuleConditionError if malformed

        rule = BusinessRule(
            name=name,
            description=description,
            rule_type=rule_type,
            condition=condition,
            priority=priority,
            created_by=actor_user_id,
        )
        rule = self.rule_repo.add(rule)
        self.audit_repo.log_action(
            user_id=actor_user_id,
            action="rule.create",
            resource=f"rule:{rule.id}",
            details={"name": name, "rule_type": rule_type, "priority": priority},
        )
        return rule

    def get_rule(self, rule_id: str) -> BusinessRule:
        rule = self.rule_repo.get_by_id(rule_id)
        if rule is None:
            raise ResourceNotFoundError(f"Rule '{rule_id}' not found.")
        return rule

    def list_rules(self, *, rule_type: str | None = None) -> list[BusinessRule]:
        return self.rule_repo.list_all_rules(rule_type=rule_type)

    def update_rule(self, rule_id: str, *, actor_user_id: str | None, **fields) -> BusinessRule:
        rule = self.get_rule(rule_id)
        if "condition" in fields and fields["condition"] is not None:
            validate_condition(fields["condition"])
        if "rule_type" in fields and fields["rule_type"] is not None and fields["rule_type"] not in VALID_RULE_TYPES:
            raise InvalidRuleTypeError(f"rule_type must be one of {VALID_RULE_TYPES}.")

        previous_snapshot = {
            "name": rule.name,
            "description": rule.description,
            "rule_type": rule.rule_type,
            "condition": rule.condition,
            "priority": rule.priority,
            "is_active": rule.is_active,
            "version": rule.version,
        }
        clean_fields = {k: v for k, v in fields.items() if v is not None}
        updated = self.rule_repo.update(rule, **clean_fields)
        self.audit_repo.log_action(
            user_id=actor_user_id,
            action="rule.update",
            resource=f"rule:{rule_id}",
            details={"previous": previous_snapshot, "new_version": updated.version},
        )
        return updated

    def deactivate_rule(self, rule_id: str, *, actor_user_id: str | None) -> BusinessRule:
        return self.update_rule(rule_id, actor_user_id=actor_user_id, is_active=False)

    def activate_rule(self, rule_id: str, *, actor_user_id: str | None) -> BusinessRule:
        return self.update_rule(rule_id, actor_user_id=actor_user_id, is_active=True)

    def evaluate(self, context: dict) -> dict:
        """Evaluates every currently-active rule against `context` (an
        applicant's raw features merged with the AI model's own
        prediction) and returns the override outcome, if any -- see
        `rules/engine.py::evaluate_rules()`. Called from the scoring
        pipeline / workflow engine after the model prediction, never
        before it: a rule like "always reject if bankruptcy_flag is set"
        should short-circuit the model's decision, but the model's
        predicted_probability is still useful CONTEXT for other rules
        (e.g. "manual review if predicted_probability is between 0.45 and
        0.55" -- a borderline-confidence rule)."""
        active_rules = [self._to_dict(r) for r in self.rule_repo.list_active()]
        result = evaluate_rules(active_rules, context)
        return {
            "has_override": result.has_override,
            "matched_rule_id": result.matched.rule_id if result.matched else None,
            "matched_rule_name": result.matched.rule_name if result.matched else None,
            "matched_rule_type": result.matched.rule_type if result.matched else None,
            "explanation": result.matched.explanation if result.matched else None,
            "evaluated_count": result.evaluated_count,
        }

    def simulate(self, contexts: list[dict], *, rule_ids: list[str] | None = None) -> list[dict]:
        """Dry-runs a set of rules (by default every currently-active rule;
        pass `rule_ids` to simulate a specific candidate set, including
        rules not yet activated) against a batch of contexts without
        persisting anything -- `POST /rules/simulate`."""
        if rule_ids:
            rules = [self._to_dict(self.get_rule(rid)) for rid in rule_ids]
        else:
            rules = [self._to_dict(r) for r in self.rule_repo.list_active()]
        return simulate_rules(rules, contexts)

    @staticmethod
    def _to_dict(rule: BusinessRule) -> dict:
        return {
            "id": rule.id,
            "name": rule.name,
            "rule_type": rule.rule_type,
            "condition": rule.condition,
            "priority": rule.priority,
            "is_active": rule.is_active,
        }
