"""Unit tests for the preprocessing building blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.feature_engineering import CreditFeatureEngineer
from preprocessing.feature_selection import FeatureSelector
from preprocessing.missing_values import MissingValueHandler
from preprocessing.outliers import OutlierWinsorizer, detect_outliers_iqr


def test_missing_value_handler_adds_indicator_and_imputes():
    df = pd.DataFrame({"income": [50_000.0, np.nan, 60_000.0, np.nan, 70_000.0]})
    handler = MissingValueHandler(numeric_columns=["income"], categorical_columns=[])
    out = handler.fit(df).transform(df)

    assert "income_was_missing" in out.columns
    assert out["income_was_missing"].tolist() == [0, 1, 0, 1, 0]
    assert out["income"].isna().sum() == 0
    # Imputed value should equal the median of the non-missing values.
    assert out.loc[1, "income"] == df["income"].median()


def test_missing_value_handler_skips_indicator_below_threshold():
    df = pd.DataFrame({"x": [1.0] * 999 + [np.nan]})  # 0.1% missing
    handler = MissingValueHandler(numeric_columns=["x"], categorical_columns=[], indicator_threshold=0.01)
    out = handler.fit(df).transform(df)
    assert "x_was_missing" not in out.columns


def test_outlier_winsorizer_clips_to_learned_bounds():
    df = pd.DataFrame({"income": list(range(1, 101))})
    winsor = OutlierWinsorizer(columns=["income"], lower_quantile=0.05, upper_quantile=0.95)
    out = winsor.fit(df).transform(df)
    assert out["income"].max() <= df["income"].quantile(0.95)
    assert out["income"].min() >= df["income"].quantile(0.05)


def test_detect_outliers_iqr_flags_extreme_values():
    values = [10] * 95 + [10_000] * 5
    df = pd.DataFrame({"loan_amount": values})
    summary = detect_outliers_iqr(df, ["loan_amount"])
    assert summary.iloc[0]["n_outliers"] == 5


def test_feature_engineer_produces_expected_ratio_columns():
    df = pd.DataFrame(
        {
            "loan_amount": [10_000.0],
            "annual_income": [50_000.0],
            "interest_rate": [10.0],
            "loan_term_months": [36],
            "savings_balance": [1000.0],
            "checking_balance": [500.0],
            "revolving_utilization": [0.4],
            "credit_score": [700],
            "num_delinquencies_2yrs": [0],
            "num_derogatory_marks": [0],
            "months_since_last_delinquency": [999],
            "num_credit_inquiries_6m": [1],
            "num_open_accounts": [5],
            "num_dependents": [0],
            "employment_length_years": [3],
            "home_ownership": ["OWN"],
            "total_credit_limit": [8000.0],
        }
    )
    engineer = CreditFeatureEngineer()
    out = engineer.fit(df).transform(df)

    assert out.loc[0, "loan_to_income_ratio"] == pytest.approx(10_000.0 / 50_000.0)
    assert out.loc[0, "is_homeowner"] == 1
    assert out.loc[0, "total_liquid_assets"] == 1500.0


def test_feature_selector_drops_zero_variance_and_correlated_columns():
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    df = pd.DataFrame(
        {
            "signal": base,
            "duplicate_of_signal": base + rng.normal(scale=0.001, size=200),  # ~perfectly correlated
            "constant": np.ones(200),
            "noise": rng.normal(size=200),
        }
    )
    y = pd.Series((base > 0).astype(int))

    selector = FeatureSelector(corr_threshold=0.95)
    out = selector.fit_transform(df, y)

    assert "constant" in selector.dropped_low_variance_
    assert len(selector.dropped_correlated_) >= 1
    assert "signal" in out.columns or "duplicate_of_signal" in out.columns
    assert not ({"signal", "duplicate_of_signal"} <= set(out.columns))


def test_feature_selector_prunes_bool_dtype_onehot_columns_too():
    """`pd.get_dummies` (used by `OneHotFrameEncoder` upstream) produces
    `bool` dtype columns. `FeatureSelector` must treat those as numeric for
    variance/correlation pruning purposes -- otherwise every one-hot column
    silently bypasses all three filters regardless of redundancy (this was a
    real bug: `X.select_dtypes(include=[np.number])` alone excludes `bool`)."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    y = pd.Series((base > 0).astype(int))
    constant_flag = pd.Series([True] * 200)  # zero-variance bool column
    duplicate_flag = pd.Series(base > 0) & pd.Series(rng.normal(size=200) > -5)  # ~perfectly correlated with y

    df = pd.DataFrame(
        {
            "signal": base,
            "constant_bool_flag": constant_flag,
            "near_duplicate_bool_flag": duplicate_flag,
        }
    )
    assert df["constant_bool_flag"].dtype == bool
    assert df["near_duplicate_bool_flag"].dtype == bool

    selector = FeatureSelector(corr_threshold=0.95)
    selector.fit_transform(df, y)

    assert "constant_bool_flag" in selector.dropped_low_variance_
