"""Tests for the data-quality validation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_quality.validators import (
    BusinessRule,
    default_credit_business_rules,
    run_full_validation,
    validate_business_rules,
    validate_dtypes,
    validate_duplicates,
    validate_leakage,
    validate_missing_values,
    validate_null_threshold,
    validate_schema,
)


def test_validate_schema_flags_out_of_range_credit_score():
    df = pd.DataFrame({"credit_score": [700, 999, 300]})  # 999 is out of the [300, 850] bound
    issues = validate_schema(df)
    assert any(i.category == "schema" and "credit_score" in i.message for i in issues)


def test_validate_schema_flags_unexpected_category():
    df = pd.DataFrame({"home_ownership": ["OWN", "SPACESHIP"]})
    issues = validate_schema(df)
    assert any("home_ownership" in i.message for i in issues)


def test_validate_missing_values_severity_scales_with_pct():
    df = pd.DataFrame({"a": [1.0] * 40 + [np.nan] * 60})  # 60% missing -> critical
    issues = validate_missing_values(df, critical_threshold=0.5)
    assert any(i.severity == "critical" for i in issues)


def test_validate_duplicates_detects_exact_dupes():
    df = pd.DataFrame({"customer_id": ["A", "B", "A"], "x": [1, 2, 1]})
    issues = validate_duplicates(df)
    assert len(issues) == 1
    assert issues[0].details["n_duplicates"] == 2


def test_validate_leakage_flags_near_perfect_correlation():
    y = pd.Series([0, 1] * 50)
    df = pd.DataFrame({"leaky_feature": y.astype(float) * 100})  # perfectly correlated with target
    issues = validate_leakage(df, y, correlation_threshold=0.95)
    assert len(issues) == 1
    assert issues[0].severity == "critical"


def test_run_full_validation_report_passed_flag():
    df = pd.DataFrame(
        {
            "age": [30, 40, 50],
            "annual_income": [50000, 60000, 70000],
            "credit_score": [700, 720, 680],
        }
    )
    report = run_full_validation(df)
    assert report.n_rows == 3
    summary = report.summary()
    assert "issues_by_severity" in summary


def test_validate_dtypes_flags_numeric_column_stored_as_string():
    df = pd.DataFrame({"annual_income": ["50000", "60,000", "70000"]})
    issues = validate_dtypes(df)
    assert any(i.category == "invalid_type" and i.severity == "critical" for i in issues)
    assert any("annual_income" in i.message for i in issues)


def test_validate_dtypes_flags_categorical_column_stored_as_numeric():
    df = pd.DataFrame({"home_ownership": [1, 2, 3]})
    issues = validate_dtypes(df)
    assert any(i.category == "invalid_type" and i.severity == "warning" for i in issues)


def test_validate_dtypes_passes_clean_dataframe():
    df = pd.DataFrame({"age": [30, 40], "home_ownership": ["OWN", "RENT"]})
    issues = validate_dtypes(df)
    assert issues == []


def test_validate_null_threshold_raises_single_critical_issue_over_budget():
    df = pd.DataFrame({"a": [1.0, None, None, None], "b": [1.0, 2.0, 3.0, 4.0]})  # a is 75% missing
    issues = validate_null_threshold(df, max_null_pct=0.3)
    assert len(issues) == 1
    assert issues[0].category == "null_threshold"
    assert issues[0].severity == "critical"
    assert issues[0].details["column"] == "a"


def test_validate_null_threshold_silent_when_under_budget():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, None]})  # 25% missing, under a 30% budget
    issues = validate_null_threshold(df, max_null_pct=0.3)
    assert issues == []


def test_default_credit_business_rules_returns_named_rules():
    rules = default_credit_business_rules()
    assert len(rules) == 3
    assert all(isinstance(r, BusinessRule) for r in rules)
    names = {r.name for r in rules}
    assert "loan_amount_vs_income" in names


def test_validate_business_rules_flags_loan_amount_vs_income_violation():
    df = pd.DataFrame(
        {
            "loan_amount": [500000, 10000],
            "annual_income": [40000, 50000],  # first row: loan is 12.5x income -> violates
            "revolving_utilization": [0.1, 0.1],
            "total_credit_limit": [1000, 1000],
            "num_delinquencies_2yrs": [0, 0],
            "months_since_last_delinquency": [99, 99],
        }
    )
    issues = validate_business_rules(df)
    assert any("loan_amount_vs_income" in i.details.get("rule", "") for i in issues)


def test_validate_business_rules_with_custom_rule_and_missing_columns_is_skipped():
    df = pd.DataFrame({"x": [1, 2, 3]})
    custom_rule = BusinessRule(
        "always_violates",
        lambda df: pd.Series([True] * len(df)),
        "always fails",
        "critical",
    )
    # A rule referencing a column that doesn't exist should be skipped, not raise.
    missing_col_rule = BusinessRule(
        "missing_column_rule",
        lambda df: df["nonexistent_column"] > 0,
        "unused",
        "warning",
    )
    issues = validate_business_rules(df, rules=[custom_rule, missing_col_rule])
    assert len(issues) == 1
    assert issues[0].details["rule"] == "always_violates"
    assert issues[0].details["n_violations"] == 3


def test_run_full_validation_accepts_business_rules_and_max_null_pct():
    df = pd.DataFrame(
        {
            "age": [30, 40, 50],
            "annual_income": [50000, 60000, 70000],
            "credit_score": [700, 720, 680],
        }
    )
    custom_rule = BusinessRule(
        "age_sanity",
        lambda df: df["age"] > 200,
        "age implausibly high",
        "warning",
    )
    report = run_full_validation(df, business_rules=[custom_rule], max_null_pct=0.1)
    summary = report.summary()
    assert summary["n_rows"] == 3
    # No violations expected from the custom rule (nobody is >200 years old here).
    assert not any(i.details.get("rule") == "age_sanity" for i in report.issues)
