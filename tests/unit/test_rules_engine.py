"""Unit tests for the pure Business Rule Engine (`rules/engine.py`) --
condition validation, single/composite condition evaluation, first-match-
wins priority ordering, and simulation. No DB, no FastAPI, no service
layer: exactly the "pure logic" surface the module's docstring promises."""

from __future__ import annotations

import pytest

from rules.engine import (
    InvalidRuleConditionError,
    evaluate_condition,
    evaluate_rules,
    simulate_rules,
    validate_condition,
)


# --- validate_condition ------------------------------------------------------
def test_validate_condition_accepts_simple_leaf():
    validate_condition({"field": "credit_score", "operator": "lt", "value": 500})


def test_validate_condition_accepts_all_composite():
    validate_condition(
        {
            "all": [
                {"field": "credit_score", "operator": "lt", "value": 500},
                {"field": "loan_amount", "operator": "gt", "value": 10000},
            ]
        }
    )


def test_validate_condition_accepts_nested_any_within_all():
    validate_condition(
        {
            "all": [
                {"field": "credit_score", "operator": "lt", "value": 500},
                {
                    "any": [
                        {"field": "state", "operator": "eq", "value": "CA"},
                        {"field": "state", "operator": "eq", "value": "NY"},
                    ]
                },
            ]
        }
    )


def test_validate_condition_rejects_missing_keys():
    with pytest.raises(InvalidRuleConditionError):
        validate_condition({"field": "credit_score", "operator": "lt"})


def test_validate_condition_rejects_unknown_operator():
    with pytest.raises(InvalidRuleConditionError):
        validate_condition({"field": "credit_score", "operator": "regex_match", "value": ".*"})


def test_validate_condition_rejects_empty_composite_list():
    with pytest.raises(InvalidRuleConditionError):
        validate_condition({"all": []})


def test_validate_condition_never_evaluates_arbitrary_code():
    """The whole point of the closed grammar: an operator name that LOOKS
    like it could be a Python builtin/eval hook is just an unknown
    operator, not an RCE vector."""
    with pytest.raises(InvalidRuleConditionError):
        validate_condition({"field": "x", "operator": "__import__", "value": "os"})


# --- evaluate_condition ------------------------------------------------------
def test_evaluate_condition_simple_lt_true():
    assert evaluate_condition({"field": "credit_score", "operator": "lt", "value": 500}, {"credit_score": 480})


def test_evaluate_condition_simple_lt_false():
    assert not evaluate_condition({"field": "credit_score", "operator": "lt", "value": 500}, {"credit_score": 700})


def test_evaluate_condition_missing_field_does_not_match_numeric_operator():
    assert not evaluate_condition({"field": "nonexistent", "operator": "lt", "value": 500}, {})


def test_evaluate_condition_all_requires_every_child():
    condition = {
        "all": [
            {"field": "credit_score", "operator": "lt", "value": 500},
            {"field": "loan_amount", "operator": "gt", "value": 10000},
        ]
    }
    assert evaluate_condition(condition, {"credit_score": 480, "loan_amount": 15000})
    assert not evaluate_condition(condition, {"credit_score": 480, "loan_amount": 5000})


def test_evaluate_condition_any_requires_one_child():
    condition = {
        "any": [
            {"field": "state", "operator": "eq", "value": "CA"},
            {"field": "state", "operator": "eq", "value": "NY"},
        ]
    }
    assert evaluate_condition(condition, {"state": "NY"})
    assert not evaluate_condition(condition, {"state": "TX"})


def test_evaluate_condition_in_and_not_in_operators():
    assert evaluate_condition({"field": "state", "operator": "in", "value": ["CA", "NY"]}, {"state": "CA"})
    assert evaluate_condition({"field": "state", "operator": "not_in", "value": ["CA", "NY"]}, {"state": "TX"})


def test_evaluate_condition_type_mismatch_does_not_crash():
    """A rule comparing a string field with a numeric operator against a
    context where the field is genuinely absent (None) must not crash the
    whole scoring request -- it just doesn't match."""
    result = evaluate_condition({"field": "home_ownership", "operator": "gt", "value": 5}, {"home_ownership": "RENT"})
    assert result is False


# --- evaluate_rules (first-match-wins, priority ordering) -------------------
def _rule(id_, name, rule_type, condition, priority=100, is_active=True):
    return {
        "id": id_,
        "name": name,
        "rule_type": rule_type,
        "condition": condition,
        "priority": priority,
        "is_active": is_active,
    }


def test_evaluate_rules_first_match_wins_by_priority():
    rules = [
        _rule(
            "r2",
            "low priority reject",
            "rejection",
            {"field": "credit_score", "operator": "lt", "value": 600},
            priority=200,
        ),
        _rule(
            "r1",
            "high priority approve",
            "approval",
            {"field": "credit_score", "operator": "lt", "value": 600},
            priority=10,
        ),
    ]
    result = evaluate_rules(rules, {"credit_score": 500})
    assert result.has_override
    assert result.matched.rule_id == "r1"
    assert len(result.all_matches) == 2  # both matched, but r1 wins on priority


def test_evaluate_rules_no_match_returns_no_override():
    rules = [_rule("r1", "reject low score", "rejection", {"field": "credit_score", "operator": "lt", "value": 300})]
    result = evaluate_rules(rules, {"credit_score": 700})
    assert not result.has_override
    assert result.matched is None
    assert result.evaluated_count == 1


def test_evaluate_rules_ignores_inactive_rules():
    rules = [
        _rule(
            "r1",
            "inactive reject",
            "rejection",
            {"field": "credit_score", "operator": "lt", "value": 900},
            is_active=False,
        )
    ]
    result = evaluate_rules(rules, {"credit_score": 500})
    assert not result.has_override
    assert result.evaluated_count == 0


def test_evaluate_rules_explanation_is_human_readable():
    rules = [_rule("r1", "reject low score", "rejection", {"field": "credit_score", "operator": "lt", "value": 500})]
    result = evaluate_rules(rules, {"credit_score": 480})
    assert "credit_score" in result.matched.explanation
    assert "480" in result.matched.explanation


# --- simulate_rules -----------------------------------------------------------
def test_simulate_rules_returns_one_result_per_context():
    rules = [_rule("r1", "reject low score", "rejection", {"field": "credit_score", "operator": "lt", "value": 500})]
    contexts = [{"credit_score": 400}, {"credit_score": 700}]
    results = simulate_rules(rules, contexts)
    assert len(results) == 2
    assert results[0]["matched_rule_id"] == "r1"
    assert results[1]["matched_rule_id"] is None
