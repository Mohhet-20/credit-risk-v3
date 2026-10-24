"""
Data validation pipeline: schema conformance, dtype conformance, missing
values, a null-rate hard gate, outliers, duplicates, cross-column business
rules, and a leakage heuristic.

Design decision: rather than adopting a full framework (Great Expectations,
pandera) for a single-table pipeline, validators are implemented as small,
composable functions returning a common `ValidationIssue` record shape. This
keeps the dependency footprint minimal and the report format trivial to
render in the Streamlit dashboard or serialize to JSON for the API, while
still covering the same categories a framework would (schema, range,
uniqueness, statistical-leakage checks).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from core.logging import get_logger
from feature_store.raw_features import SCHEMA
from preprocessing.outliers import detect_outliers_iqr

logger = get_logger(__name__)


@dataclass
class ValidationIssue:
    category: str  # schema | invalid_type | missing | null_threshold | outlier | duplicate | business_rule | leakage
    severity: str  # info | warning | critical
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    n_rows: int = 0
    n_columns: int = 0

    @property
    def passed(self) -> bool:
        return not any(i.severity == "critical" for i in self.issues)

    def summary(self) -> dict:
        return {
            "passed": self.passed,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_issues": len(self.issues),
            "issues_by_severity": {
                sev: sum(1 for i in self.issues if i.severity == sev) for sev in ("info", "warning", "critical")
            },
            "issues": [
                {"category": i.category, "severity": i.severity, "message": i.message, "details": i.details}
                for i in self.issues
            ],
        }


def validate_schema(df: pd.DataFrame) -> list[ValidationIssue]:
    issues = []
    expected = set(SCHEMA.all_features)
    actual = set(df.columns)
    missing_cols = expected - actual
    if missing_cols:
        issues.append(ValidationIssue("schema", "critical", f"Missing expected columns: {sorted(missing_cols)}"))

    for col, (lo, hi) in SCHEMA.RANGE_BOUNDS.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]
        if len(out_of_range) > 0:
            issues.append(
                ValidationIssue(
                    "schema",
                    "warning",
                    f"{len(out_of_range)} rows have {col} outside expected range [{lo}, {hi}]",
                    {"column": col, "n_violations": len(out_of_range)},
                )
            )

    for col, allowed in SCHEMA.CATEGORICAL_ALLOWED_VALUES.items():
        if col not in df.columns:
            continue
        bad_values = set(df[col].dropna().unique()) - allowed
        if bad_values:
            issues.append(
                ValidationIssue(
                    "schema",
                    "warning",
                    f"{col} contains unexpected categories: {sorted(bad_values)}",
                    {"column": col, "unexpected_values": sorted(bad_values)},
                )
            )
    return issues


def validate_dtypes(df: pd.DataFrame) -> list[ValidationIssue]:
    """Flags columns whose *actual* pandas dtype doesn't match what the raw
    feature schema expects (numeric columns must be numeric-typed,
    categorical columns must be string/object-typed) -- catches the "a CSV
    import silently parsed a numeric column as strings because of a stray
    comma" class of bug before it ever reaches the feature pipeline, which
    would otherwise fail with a much less obvious error deeper in
    `OneHotEncoder`/`OutlierWinsorizer`."""
    issues = []
    for col in SCHEMA.numeric:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(
                ValidationIssue(
                    "invalid_type",
                    "critical",
                    f"{col} is expected to be numeric but has dtype {df[col].dtype}",
                    {"column": col, "actual_dtype": str(df[col].dtype), "expected": "numeric"},
                )
            )
    for col in SCHEMA.categorical:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            issues.append(
                ValidationIssue(
                    "invalid_type",
                    "warning",
                    f"{col} is expected to be categorical (string) but has dtype {df[col].dtype}",
                    {"column": col, "actual_dtype": str(df[col].dtype), "expected": "categorical"},
                )
            )
    return issues


def validate_missing_values(df: pd.DataFrame, critical_threshold: float = 0.5) -> list[ValidationIssue]:
    issues = []
    missing_pct = df.isna().mean()
    for col, pct in missing_pct.items():
        if pct == 0:
            continue
        severity = "critical" if pct > critical_threshold else ("warning" if pct > 0.1 else "info")
        issues.append(
            ValidationIssue(
                "missing", severity, f"{col} is {pct:.1%} missing", {"column": col, "pct_missing": round(float(pct), 4)}
            )
        )
    return issues


def validate_null_threshold(df: pd.DataFrame, max_null_pct: float = 0.3) -> list[ValidationIssue]:
    """A single explicit "hard gate" check -- distinct from `validate_missing_values`
    (which reports every column's missing rate at graduated severity): this
    raises exactly one critical issue if the WORST column exceeds
    `max_null_pct`, intended as a simple boolean gate a caller can check
    without inspecting every column ("did anything blow past our null
    budget, yes or no")."""
    missing_pct = df.isna().mean()
    worst_col = missing_pct.idxmax() if len(missing_pct) else None
    worst_pct = float(missing_pct.max()) if len(missing_pct) else 0.0
    if worst_col is None or worst_pct <= max_null_pct:
        return []
    return [
        ValidationIssue(
            "null_threshold",
            "critical",
            f"{worst_col} exceeds the {max_null_pct:.0%} null-rate threshold ({worst_pct:.1%} missing)",
            {"column": worst_col, "pct_missing": round(worst_pct, 4), "threshold": max_null_pct},
        )
    ]


def validate_outliers(df: pd.DataFrame, columns: list[str], pct_threshold: float = 0.05) -> list[ValidationIssue]:
    outlier_summary = detect_outliers_iqr(df, columns)
    issues = []
    for _, row in outlier_summary.iterrows():
        if row["pct_outliers"] > pct_threshold * 100:
            issues.append(
                ValidationIssue(
                    "outlier",
                    "warning",
                    f"{row['column']} has {row['pct_outliers']:.2f}% IQR outliers",
                    {"column": row["column"], "pct_outliers": row["pct_outliers"]},
                )
            )
    return issues


def validate_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> list[ValidationIssue]:
    dupe_mask = df.duplicated(subset=subset, keep=False)
    n_dupes = int(dupe_mask.sum())
    if n_dupes == 0:
        return []
    return [
        ValidationIssue(
            "duplicate",
            "warning" if n_dupes < len(df) * 0.01 else "critical",
            f"{n_dupes} duplicate rows detected" + (f" (subset={subset})" if subset else ""),
            {"n_duplicates": n_dupes},
        )
    ]


class BusinessRule(NamedTuple):
    """A named, severity-tagged predicate over the whole DataFrame. The
    predicate returns the *violating* row mask (True = violates the rule),
    not a pass/fail boolean -- this lets `validate_business_rules` report how
    many rows failed, not just whether any did."""

    name: str
    predicate: Callable[[pd.DataFrame], pd.Series]
    message: str
    severity: str = "warning"


def default_credit_business_rules() -> list[BusinessRule]:
    """Domain-specific sanity rules for credit applications that go beyond
    per-column range checks (`validate_schema`) -- these check relationships
    BETWEEN columns, which is where most real-world "technically valid but
    nonsensical" records hide."""
    return [
        BusinessRule(
            "loan_amount_vs_income",
            lambda df: (df["loan_amount"] > df["annual_income"] * 10),
            "loan_amount exceeds 10x annual_income",
            "warning",
        ),
        BusinessRule(
            "positive_credit_limit_utilization",
            lambda df: (df["revolving_utilization"] > 0) & (df["total_credit_limit"] <= 0),
            "revolving_utilization > 0 but total_credit_limit is not positive",
            "critical",
        ),
        BusinessRule(
            "delinquency_recency_consistency",
            lambda df: (df["num_delinquencies_2yrs"] == 0) & (df["months_since_last_delinquency"] < 24),
            "months_since_last_delinquency is recent but num_delinquencies_2yrs is zero",
            "warning",
        ),
    ]


def validate_business_rules(df: pd.DataFrame, rules: list[BusinessRule] | None = None) -> list[ValidationIssue]:
    rules = rules if rules is not None else default_credit_business_rules()
    issues = []
    for rule in rules:
        try:
            violations = rule.predicate(df)
        except (KeyError, TypeError):
            continue
        n_violations = int(violations.sum())
        if n_violations == 0:
            continue
        issues.append(
            ValidationIssue(
                "business_rule",
                rule.severity,
                f"[{rule.name}] {rule.message} ({n_violations} row(s))",
                {"rule": rule.name, "n_violations": n_violations},
            )
        )
    return issues


def validate_leakage(df: pd.DataFrame, target: pd.Series, correlation_threshold: float = 0.98) -> list[ValidationIssue]:
    """Flags numeric features suspiciously perfectly correlated with the target —
    a common symptom of a feature that's actually a post-outcome artifact."""
    issues = []
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        if numeric_df[col].nunique() <= 1:
            continue
        corr = abs(np.corrcoef(numeric_df[col].fillna(numeric_df[col].median()), target)[0, 1])
        if corr > correlation_threshold:
            issues.append(
                ValidationIssue(
                    "leakage",
                    "critical",
                    f"{col} is suspiciously correlated with target (r={corr:.3f})",
                    {"column": col, "correlation": round(float(corr), 4)},
                )
            )
    return issues


def run_full_validation(
    df: pd.DataFrame,
    target: pd.Series | None = None,
    outlier_columns: list[str] | None = None,
    duplicate_subset: list[str] | None = None,
    business_rules: list[BusinessRule] | None = None,
    max_null_pct: float = 0.3,
) -> ValidationReport:
    report = ValidationReport(n_rows=len(df), n_columns=df.shape[1])
    report.issues.extend(validate_schema(df))
    report.issues.extend(validate_dtypes(df))
    report.issues.extend(validate_missing_values(df))
    report.issues.extend(validate_null_threshold(df, max_null_pct=max_null_pct))
    if outlier_columns:
        report.issues.extend(validate_outliers(df, outlier_columns))
    report.issues.extend(validate_duplicates(df, subset=duplicate_subset))
    report.issues.extend(validate_business_rules(df, rules=business_rules))
    if target is not None:
        report.issues.extend(validate_leakage(df, target))

    logger.info("Data validation complete", extra={"context": report.summary()["issues_by_severity"]})
    return report
