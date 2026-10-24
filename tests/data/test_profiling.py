"""Tests for the data profiling module (`data_quality/profiling.py`) --
descriptive statistics, deliberately separate from validation's pass/fail
verdicts (see the module docstring for the design rationale)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_quality.profiling import generate_profile


def test_generate_profile_returns_all_expected_sections():
    df = pd.DataFrame(
        {
            "age": [30, 40, 50, 60],
            "annual_income": [50000, 60000, 70000, 80000],
            "home_ownership": ["OWN", "RENT", "OWN", "MORTGAGE"],
        }
    )
    profile = generate_profile(df)
    for key in (
        "n_rows",
        "n_columns",
        "column_statistics",
        "correlations",
        "missing_value_summary",
        "duplicate_analysis",
        "outlier_summary",
        "cardinality",
        "business_summary",
    ):
        assert key in profile
    assert profile["n_rows"] == 4
    assert profile["n_columns"] == 3


def test_generate_profile_column_statistics_numeric_vs_categorical():
    df = pd.DataFrame({"age": [30, 40, 50], "home_ownership": ["OWN", "RENT", "OWN"]})
    profile = generate_profile(df)
    stats_by_col = {s["column"]: s for s in profile["column_statistics"]}

    age_stats = stats_by_col["age"]
    assert "mean" in age_stats and "median" in age_stats and "std" in age_stats

    ownership_stats = stats_by_col["home_ownership"]
    assert "top_values" in ownership_stats
    assert ownership_stats["top_values"][0]["value"] == "OWN"
    assert ownership_stats["top_values"][0]["count"] == 2


def test_generate_profile_missing_value_summary_reports_worst_columns():
    df = pd.DataFrame({"a": [1.0, None, None], "b": [1.0, 2.0, 3.0]})
    profile = generate_profile(df)
    summary = profile["missing_value_summary"]
    assert summary["n_columns_with_missing"] == 1
    assert summary["worst_columns"][0]["column"] == "a"
    assert summary["worst_columns"][0]["pct_missing"] > 0.6


def test_generate_profile_duplicate_analysis_counts_groups():
    df = pd.DataFrame({"x": [1, 1, 2, 3], "y": ["a", "a", "b", "c"]})
    profile = generate_profile(df)
    dup = profile["duplicate_analysis"]
    assert dup["n_exact_duplicate_rows"] == 2
    assert dup["n_distinct_duplicate_groups"] == 1


def test_generate_profile_cardinality_flags_high_cardinality_categorical():
    df = pd.DataFrame({"customer_id": [f"CUST-{i}" for i in range(60)]})
    profile = generate_profile(df)
    card = {c["column"]: c for c in profile["cardinality"]}
    assert card["customer_id"]["is_high_cardinality"] is True


def test_generate_profile_cardinality_does_not_flag_high_cardinality_numeric():
    df = pd.DataFrame({"annual_income": np.arange(60) * 1000})
    profile = generate_profile(df)
    card = {c["column"]: c for c in profile["cardinality"]}
    # Numeric columns are never flagged high-cardinality even with many
    # unique values -- that check is categorical-only by design.
    assert card["annual_income"]["is_high_cardinality"] is False


def test_generate_profile_correlations_returns_top_pairs():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],  # perfectly correlated with a
            "c": [5, 3, 1, -1, -3],  # perfectly anti-correlated with a
        }
    )
    profile = generate_profile(df)
    correlations = profile["correlations"]
    assert len(correlations) == 3  # 3 numeric columns -> 3 pairs
    top = correlations[0]
    assert abs(top["correlation"]) > 0.99


def test_generate_profile_business_summary_includes_target_rate():
    df = pd.DataFrame({"annual_income": [50000, 60000, 70000, 80000]})
    target = pd.Series([0, 1, 0, 1])
    profile = generate_profile(df, target=target)
    business = profile["business_summary"]
    assert business["target_positive_rate"] == 0.5
    assert business["annual_income_median"] == 65000.0
