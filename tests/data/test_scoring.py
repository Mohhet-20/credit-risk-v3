"""Tests for the composite data-quality score (`data_quality/scoring.py`)."""

from __future__ import annotations

import time

import pandas as pd

from data_quality.profiling import generate_profile
from data_quality.scoring import compute_quality_score
from data_quality.validators import run_full_validation


def test_compute_quality_score_perfect_dataset_scores_100():
    df = pd.DataFrame(
        {
            "age": [30, 40, 50],
            "annual_income": [50000, 60000, 70000],
            "credit_score": [700, 720, 680],
        }
    )
    validation = run_full_validation(df)
    profile = generate_profile(df)
    score = compute_quality_score(validation, profile)

    assert score["completeness_score"] == 100.0
    assert score["uniqueness_score"] == 100.0
    assert score["overall_score"] == round((score["validity_score"] + 100 + 100) / 3, 2)
    assert score["freshness_hours"] is None  # no dataset_path given
    assert score["n_validation_issues"] == len(validation.issues)


def test_compute_quality_score_penalizes_missing_values():
    df = pd.DataFrame({"a": [1.0, None, None, 4.0]})  # 50% missing
    validation = run_full_validation(df)
    profile = generate_profile(df)
    score = compute_quality_score(validation, profile)
    assert score["completeness_score"] < 100.0


def test_compute_quality_score_penalizes_duplicates():
    df = pd.DataFrame({"x": [1, 1, 1, 2]})
    validation = run_full_validation(df)
    profile = generate_profile(df)
    score = compute_quality_score(validation, profile)
    assert score["uniqueness_score"] < 100.0


def test_compute_quality_score_critical_issue_lowers_validity_more_than_warning():
    clean_df = pd.DataFrame({"age": [30, 40, 50]})
    clean_validation = run_full_validation(clean_df)
    clean_profile = generate_profile(clean_df)
    clean_score = compute_quality_score(clean_validation, clean_profile)

    # 60% missing on a single column triggers a "critical" missing-value issue.
    dirty_df = pd.DataFrame({"age": [30] + [None] * 9})
    dirty_validation = run_full_validation(dirty_df)
    dirty_profile = generate_profile(dirty_df)
    dirty_score = compute_quality_score(dirty_validation, dirty_profile)

    assert dirty_score["validity_score"] < clean_score["validity_score"]
    assert dirty_score["validation_passed"] is False


def test_compute_quality_score_freshness_reported_when_dataset_path_given(tmp_path):
    df = pd.DataFrame({"age": [30, 40, 50]})
    path = tmp_path / "reference.csv"
    df.to_csv(path, index=False)
    time.sleep(0.05)

    validation = run_full_validation(df)
    profile = generate_profile(df)
    score = compute_quality_score(validation, profile, dataset_path=path)

    assert score["freshness_hours"] is not None
    assert score["freshness_hours"] >= 0.0


def test_compute_quality_score_freshness_none_for_nonexistent_path():
    df = pd.DataFrame({"age": [30, 40, 50]})
    validation = run_full_validation(df)
    profile = generate_profile(df)
    score = compute_quality_score(validation, profile, dataset_path="/no/such/file.csv")
    assert score["freshness_hours"] is None


def test_compute_quality_score_overall_never_below_zero():
    # Many critical issues shouldn't push the score negative.
    df = pd.DataFrame({"a": [None] * 20})
    validation = run_full_validation(df, max_null_pct=0.01)
    profile = generate_profile(df)
    score = compute_quality_score(validation, profile)
    assert score["validity_score"] >= 0.0
    assert score["overall_score"] >= 0.0
