"""Tests for the PSI-based drift detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_quality.drift_detection import compute_categorical_psi, compute_numeric_psi, run_drift_detection


def test_numeric_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, 1000))
    current = pd.Series(rng.normal(0, 1, 1000))
    psi = compute_numeric_psi(reference, current)
    assert psi < 0.1


def test_numeric_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, 1000))
    current = pd.Series(rng.normal(5, 1, 1000))  # large mean shift
    psi = compute_numeric_psi(reference, current)
    assert psi > 0.25


def test_categorical_psi_zero_for_identical_proportions():
    reference = pd.Series(["A"] * 500 + ["B"] * 500)
    current = pd.Series(["A"] * 500 + ["B"] * 500)
    psi = compute_categorical_psi(reference, current)
    assert psi == 0.0


def test_categorical_psi_positive_for_shifted_proportions():
    reference = pd.Series(["A"] * 900 + ["B"] * 100)
    current = pd.Series(["A"] * 100 + ["B"] * 900)
    psi = compute_categorical_psi(reference, current)
    assert psi > 0.25


def test_run_drift_detection_overall_severity_none_for_stable_data():
    rng = np.random.default_rng(0)
    reference_df = pd.DataFrame(
        {
            "income": rng.normal(50000, 5000, 500),
            "state": rng.choice(["CA", "TX"], 500),
        }
    )
    current_df = pd.DataFrame(
        {
            "income": rng.normal(50000, 5000, 500),
            "state": rng.choice(["CA", "TX"], 500),
        }
    )
    report = run_drift_detection(reference_df, current_df, ["income"], ["state"])
    assert report.overall_severity in ("none", "moderate")  # random noise shouldn't trigger "significant"
