"""
Tests for the credit-risk validation and fair-lending layer: KS/Gini/decile
metrics, scorecard scaling, adverse-action reason codes, disparate-impact
analysis, monotone constraints, protected-attribute exclusion and the
out-of-time split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.fairness import disparate_impact_report
from core.constants import MODEL_EXCLUDED_FEATURES, TARGET_COLUMN
from core.scorecard import BASE_ODDS, BASE_SCORE, PDO, pd_to_score, score_to_pd
from explainability.adverse_action import MAX_REASONS, adverse_action_reasons, reason_group_for
from pipelines.run_training import _out_of_time_split, _select_champion
from training.credit_metrics import credit_validation_metrics, decile_table, gini, ks_statistic
from training.tuning import MONOTONE_DIRECTIONS, build_model


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.15, 4000)
    p = np.clip(0.15 + 0.25 * (y - 0.15) + rng.normal(0, 0.1, 4000), 0.001, 0.999)
    return y, p


# --- credit metrics -----------------------------------------------------------


def test_gini_and_ks_bounds_and_perfect_separation(scored):
    y, p = scored
    assert 0 < gini(y, p) < 1
    assert 0 < ks_statistic(y, p) < 1
    assert ks_statistic(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert gini(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_decile_table_ranks_riskiest_first_and_captures_all_defaults(scored):
    y, p = scored
    table = decile_table(y, p)
    assert len(table) == 10
    assert table["n"].sum() == len(y)
    assert table["capture_rate"].iloc[-1] == pytest.approx(1.0)
    assert table["bad_rate"].iloc[0] > table["bad_rate"].iloc[-1]
    assert table["mean_pd"].is_monotonic_decreasing


def test_validation_metrics_include_confidence_intervals(scored):
    y, p = scored
    m = credit_validation_metrics(y, p)
    lo, hi = m["roc_auc_ci95"]
    assert lo <= m["roc_auc"] <= hi
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1, abs=1e-3)
    assert m["n"] == len(y) and m["n_defaults"] == int(y.sum())


# --- scorecard ----------------------------------------------------------------


def test_scorecard_anchor_and_points_to_double_odds():
    base_pd = 1 / (1 + BASE_ODDS)
    assert pd_to_score(base_pd) == BASE_SCORE
    doubled_pd = 1 / (1 + 2 * BASE_ODDS)
    assert pd_to_score(doubled_pd) == BASE_SCORE + PDO
    assert score_to_pd(BASE_SCORE) == pytest.approx(base_pd)


def test_scorecard_is_monotone_and_clipped():
    pds = np.linspace(0.0, 1.0, 50)
    scores = [pd_to_score(p) for p in pds]
    assert all(a >= b for a, b in zip(scores, scores[1:], strict=False))
    assert min(scores) >= 300 and max(scores) <= 850


# --- adverse action -----------------------------------------------------------


def test_reason_groups_cover_engineered_and_encoded_features():
    assert reason_group_for("credit_score").code == "R01"
    assert reason_group_for("credit_score_band_poor").code == "R01"
    assert reason_group_for("utilization_x_low_score").code == "R02"
    assert reason_group_for("home_ownership_RENT").code == "R13"
    assert reason_group_for("loan_purpose_target_enc").code == "R14"
    assert reason_group_for("something_new").code == "R99"


def test_adverse_action_aggregates_groups_and_ignores_favourable_factors():
    contributions = [
        {"feature": "revolving_utilization", "shap_value": 0.20, "feature_value": 0.9},
        {"feature": "utilization_x_low_score", "shap_value": 0.15, "feature_value": 0.9},
        {"feature": "credit_score", "shap_value": 0.30, "feature_value": 560},
        {"feature": "num_derogatory_marks", "shap_value": 0.05, "feature_value": 1},
        {"feature": "annual_income", "shap_value": -0.40, "feature_value": 250000},
        {"feature": "debt_to_income_ratio", "shap_value": 0.02, "feature_value": 30},
        {"feature": "num_credit_inquiries_6m", "shap_value": 0.01, "feature_value": 3},
    ]
    reasons = adverse_action_reasons(contributions)
    assert len(reasons) == MAX_REASONS
    # utilization (0.20 + 0.15) outranks credit score (0.30) once grouped.
    assert [r["code"] for r in reasons[:2]] == ["R02", "R01"]
    assert set(reasons[0]["features"]) == {"revolving_utilization", "utilization_x_low_score"}
    assert all(r["contribution"] > 0 for r in reasons)
    assert "R08" not in {r["code"] for r in reasons}  # income helped; never a decline reason


def test_adverse_action_empty_when_nothing_increased_risk():
    assert adverse_action_reasons([{"feature": "credit_score", "shap_value": -0.1, "feature_value": 800}]) == []


# --- fairness -----------------------------------------------------------------


def test_disparate_impact_flags_four_fifths_violation():
    n = 300
    raw = pd.DataFrame({"age": [25] * n + [70] * n, "marital_status": ["single"] * n + ["married"] * n})
    y = np.r_[np.zeros(n - 30), np.ones(30), np.zeros(n - 30), np.ones(30)]
    p = np.r_[np.full(n, 0.10), np.r_[np.full(n // 2, 0.10), np.full(n // 2, 0.90)]]
    report = disparate_impact_report(raw, y, p, threshold=0.5)
    age = report["attributes"]["age_band"]
    assert not age["passes"]
    assert age["min_adverse_impact_ratio"] == pytest.approx(0.5)
    assert report["passes"] is False


def test_disparate_impact_passes_for_equal_treatment(scored):
    y, p = scored
    rng = np.random.default_rng(1)
    raw = pd.DataFrame(
        {"age": rng.integers(18, 80, len(y)), "marital_status": rng.choice(["single", "married"], len(y))}
    )
    report = disparate_impact_report(raw, y, p, threshold=0.5)
    assert report["passes"]


# --- model construction / pipeline -------------------------------------------


def test_monotone_constraints_follow_feature_order():
    features = ["credit_score", "loan_purpose_target_enc", "debt_to_income_ratio"]
    xgb = build_model("xgboost", {}, feature_names=features)
    assert xgb.get_params()["monotone_constraints"] == "(-1,0,1)"
    lgbm = build_model("lightgbm", {}, feature_names=features)
    assert lgbm.get_params()["monotone_constraints"] == [-1, 0, 1]
    cat = build_model("catboost", {}, feature_names=features)
    assert cat.get_params()["monotone_constraints"] == "(-1,0,1)"
    from sklearn.base import clone

    for model in (xgb, lgbm, cat):
        clone(model)  # StackingClassifier clones every base estimator
    assert MONOTONE_DIRECTIONS["credit_score"] == -1


@pytest.mark.parametrize(
    "model_name,params",
    [("lightgbm", {"n_estimators": 50}), ("xgboost", {"n_estimators": 50}), ("catboost", {"iterations": 50})],
)
def test_monotone_constraint_holds_for_trained_model(transformed_data, model_name, params):
    Xt, y = transformed_data
    model = build_model(model_name, params, feature_names=list(Xt.columns)).fit(Xt, y)
    probe = pd.concat([Xt.iloc[[0]]] * 20, ignore_index=True)
    probe["credit_score"] = np.linspace(Xt["credit_score"].min(), Xt["credit_score"].max(), 20)
    proba = model.predict_proba(probe)[:, 1]
    assert np.all(np.diff(proba) <= 1e-12)


def test_monotone_in_raw_credit_score_through_full_pipeline(fitted_feature_pipeline, raw_dataset_no_leak):
    """The property that matters to an applicant: raising the *raw* bureau
    score never raises PD, end to end through feature engineering. Derived
    copies of the score (e.g. band one-hots) must not reopen the door."""
    X, y = raw_dataset_no_leak
    Xt = fitted_feature_pipeline.transform(X)
    model = build_model("catboost", {"iterations": 80}, feature_names=list(Xt.columns)).fit(Xt, y)
    for i in range(5):
        sweep = pd.concat([X.iloc[[i]]] * 36, ignore_index=True)
        sweep["credit_score"] = np.arange(500, 860, 10)
        proba = model.predict_proba(fitted_feature_pipeline.transform(sweep))[:, 1]
        assert np.all(np.diff(proba) <= 1e-12), proba


def test_descriptive_bands_are_not_model_features(fitted_feature_pipeline):
    assert not any("_band" in f for f in fitted_feature_pipeline.feature_names)


def test_protected_attributes_never_reach_the_model(fitted_feature_pipeline):
    for name in fitted_feature_pipeline.feature_names:
        for protected in MODEL_EXCLUDED_FEATURES:
            assert name != protected and not name.startswith(protected + "_")


def test_protected_attributes_do_not_change_predictions(fitted_feature_pipeline, raw_dataset_no_leak):
    X, _ = raw_dataset_no_leak
    row = X.iloc[[0]].copy()
    altered = row.copy()
    altered["age"] = 79
    altered["marital_status"] = "widowed"
    altered["state"] = "TX" if row["state"].iloc[0] != "TX" else "CA"
    pd.testing.assert_frame_equal(fitted_feature_pipeline.transform(row), fitted_feature_pipeline.transform(altered))


def test_out_of_time_split_is_chronological(raw_dataset):
    df = raw_dataset.copy()
    df["application_date"] = pd.to_datetime(df["application_date"])
    train, valid, test = _out_of_time_split(df)
    assert len(train) + len(valid) + len(test) == len(df)
    assert train["application_date"].max() <= valid["application_date"].min()
    assert valid["application_date"].max() <= test["application_date"].min()
    assert TARGET_COLUMN in test.columns


def test_parsimony_rule_prefers_single_model_unless_ensemble_clearly_better():
    rows = [
        {"model": "lightgbm", "pr_auc": 0.500},
        {"model": "xgboost", "pr_auc": 0.490},
        {"model": "blending_ensemble", "pr_auc": 0.503},
    ]
    assert _select_champion(rows, "pr_auc", min_improvement=0.005) == "lightgbm"
    rows[2]["pr_auc"] = 0.520
    assert _select_champion(rows, "pr_auc", min_improvement=0.005) == "blending_ensemble"
