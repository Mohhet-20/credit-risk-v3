"""
Business Rule Engine: evaluates admin-configured rules (`database.models.
BusinessRule`) against a scoring context (the applicant's raw features plus
the AI model's own prediction) to optionally OVERRIDE the model's decision
with a deterministic approve/reject/manual-review outcome -- e.g. "always
manually review any application with a loan amount over $100,000,
regardless of what the model predicts" or "always reject if there's an
open bankruptcy flag" are policy rules a risk team needs to enforce
immediately without waiting for a model retrain.

Design decision: this module is pure and has ZERO dependencies on FastAPI,
the database, or any service -- `evaluate_rules()` takes plain
`list[dict]` rules and a plain `dict` context, returns a plain result. This
is what makes `POST /rules/simulate` (dry-run a candidate rule set against
historical/synthetic data before activating it) trivial: simulation calls
the exact same function real scoring does, just with `persist=False`
implied by the caller never writing the result anywhere.

Design decision: conditions are a small closed JSON grammar (field/
operator/value, or `{"all": [...]}`/`{"any": [...]}` composites of the
same) -- NOT a free-text expression string ever passed to `eval()` or
similar. A rule is DATA, never CODE. This is a hard security boundary
(an admin-editable rule must never become a remote-code-execution vector)
and also what makes every rule mechanically explainable ("this rule fired
because credit_score (490) < 500") rather than an opaque expression only
its author can reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_OPERATORS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a is not None and a < b,
    "lte": lambda a, b: a is not None and a <= b,
    "gt": lambda a, b: a is not None and a > b,
    "gte": lambda a, b: a is not None and a >= b,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
    "contains": lambda a, b: b in a if a is not None else False,
}

VALID_RULE_TYPES = ("approval", "rejection", "manual_review")


class InvalidRuleConditionError(ValueError):
    """Raised when a rule's `condition` JSON doesn't match the supported
    grammar -- e.g. an unknown operator, or a leaf condition missing
    `field`/`operator`/`value`. Raised at rule-CREATE time (see
    `services/rule_service.py`) so a malformed rule is rejected before it
    can ever silently fail to fire during real scoring."""


def validate_condition(condition: dict) -> None:
    """Recursively validates a condition tree without evaluating it against
    any data -- used to reject a malformed rule at creation/update time."""
    if "all" in condition or "any" in condition:
        key = "all" if "all" in condition else "any"
        children = condition[key]
        if not isinstance(children, list) or not children:
            raise InvalidRuleConditionError(f"'{key}' must be a non-empty list of sub-conditions.")
        for child in children:
            validate_condition(child)
        return

    missing = {"field", "operator", "value"} - condition.keys()
    if missing:
        raise InvalidRuleConditionError(f"Condition missing required key(s): {sorted(missing)}")
    if condition["operator"] not in _OPERATORS:
        raise InvalidRuleConditionError(
            f"Unknown operator '{condition['operator']}'. Valid operators: {sorted(_OPERATORS)}"
        )


def evaluate_condition(condition: dict, context: dict[str, Any]) -> bool:
    """Evaluates a single condition (or composite `all`/`any` tree) against
    a flat context dict. Missing context fields evaluate to `None`, which
    every comparison operator except `eq`/`ne`/`in`/`not_in` treats as
    non-matching (a missing field should never accidentally satisfy a
    `gt`/`lt` numeric comparison)."""
    if "all" in condition:
        return all(evaluate_condition(c, context) for c in condition["all"])
    if "any" in condition:
        return any(evaluate_condition(c, context) for c in condition["any"])

    value = context.get(condition["field"])
    operator_fn = _OPERATORS[condition["operator"]]
    try:
        return bool(operator_fn(value, condition["value"]))
    except TypeError:
        # e.g. comparing None < 500, or a str field against a numeric operator
        # on a context that doesn't have that field yet -- a rule that can't
        # be evaluated against this context simply doesn't match, rather than
        # crashing the whole scoring request over one misconfigured rule.
        return False


@dataclass
class RuleMatch:
    rule_id: str
    rule_name: str
    rule_type: str
    priority: int
    explanation: str


@dataclass
class RuleEvaluationResult:
    matched: RuleMatch | None
    evaluated_count: int
    all_matches: list[RuleMatch] = field(default_factory=list)

    @property
    def has_override(self) -> bool:
        return self.matched is not None


def evaluate_rules(rules: list[dict], context: dict[str, Any]) -> RuleEvaluationResult:
    """Evaluates `rules` (each a dict with at least `id`, `name`,
    `rule_type`, `condition`, `priority`, `is_active`) against `context` in
    ascending priority order (lower number = runs first = higher
    precedence) and returns the FIRST match as the override decision --
    first-match-wins, not "most specific wins" or "all matches combined",
    because a deterministic, explainable "which single rule decided this"
    answer matters more for an auditable credit decision than a cleverer
    conflict-resolution scheme would. Every match (not just the winning
    one) is still recorded in `all_matches` so an admin reviewing a
    decision can see every rule that WOULD have fired, not just the one
    that did -- useful for spotting overlapping/redundant rules."""
    active = sorted((r for r in rules if r.get("is_active", True)), key=lambda r: r["priority"])
    all_matches: list[RuleMatch] = []
    winner: RuleMatch | None = None

    for rule in active:
        if evaluate_condition(rule["condition"], context):
            match = RuleMatch(
                rule_id=rule["id"],
                rule_name=rule["name"],
                rule_type=rule["rule_type"],
                priority=rule["priority"],
                explanation=_explain(rule["condition"], context),
            )
            all_matches.append(match)
            if winner is None:
                winner = match

    return RuleEvaluationResult(matched=winner, evaluated_count=len(active), all_matches=all_matches)


def simulate_rules(rules: list[dict], contexts: list[dict[str, Any]]) -> list[dict]:
    """Dry-runs `rules` against a batch of contexts (real historical
    predictions, or synthetic data -- see `data/generate_synthetic_data.py`)
    without persisting anything, for `POST /rules/simulate`. Returns one
    summary dict per context so a risk analyst can see, before activating a
    candidate rule set, what fraction of a representative population each
    rule would affect and whether any rules never fire at all (dead rules)."""
    results = []
    for context in contexts:
        outcome = evaluate_rules(rules, context)
        results.append(
            {
                "context": context,
                "matched_rule_id": outcome.matched.rule_id if outcome.matched else None,
                "matched_rule_type": outcome.matched.rule_type if outcome.matched else None,
                "explanation": outcome.matched.explanation if outcome.matched else None,
            }
        )
    return results


def _explain(condition: dict, context: dict[str, Any], depth: int = 0) -> str:
    if "all" in condition:
        return "(" + " AND ".join(_explain(c, context) for c in condition["all"]) + ")"
    if "any" in condition:
        return "(" + " OR ".join(_explain(c, context) for c in condition["any"]) + ")"
    field_name = condition["field"]
    actual = context.get(field_name)
    return f"{field_name} ({actual!r}) {condition['operator']} {condition['value']!r}"
