"""Unit tests for calibration, threshold optimization, and cross-validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from training.calibration import calibration_metrics, compute_calibration_curve
from training.cross_validation import run_stratified_cv
from training.threshold_optimization import evaluate_at_threshold, optimize_threshold
from training.tuning import build_model


def test_optimize_threshold_cost_sensitive_favors_recall_when_fn_costly():
    rng = np.random.default_rng(0)
    y_true = pd.Series([0] * 90 + [1] * 10)
    y_proba = np.concatenate([rng.uniform(0, 0.4, 90), rng.uniform(0.3, 0.9, 10)])

    result_high_fn_cost = optimize_threshold(y_true, y_proba, method="cost_sensitive", cost_fn=20.0, cost_fp=1.0)
    result_low_fn_cost = optimize_threshold(y_true, y_proba, method="cost_sensitive", cost_fn=1.0, cost_fp=1.0)

    # A much higher false-negative cost should push the threshold DOWN (catch more positives).
    assert result_high_fn_cost.threshold <= result_low_fn_cost.threshold


def test_optimize_threshold_f1_returns_valid_range():
    y_true = pd.Series([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.4, 0.6, 0.05, 0.95])
    result = optimize_threshold(y_true, y_proba, method="f1")
    assert 0.0 <= result.threshold <= 1.0


def test_evaluate_at_threshold_confusion_counts_sum_to_n():
    y_true = pd.Series([0, 1, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.8])
    metrics = evaluate_at_threshold(y_true, y_proba, threshold=0.5)
    total = (
        metrics["true_positives"] + metrics["true_negatives"] + metrics["false_positives"] + metrics["false_negatives"]
    )
    assert total == len(y_true)


def test_calibration_metrics_perfect_predictions_have_zero_brier():
    y_true = pd.Series([0, 1, 0, 1])
    y_proba = np.array([0.0, 1.0, 0.0, 1.0])
    metrics = calibration_metrics(y_true, y_proba)
    assert metrics["brier_score"] == 0.0


def test_calibration_curve_returns_expected_columns():
    rng = np.random.default_rng(1)
    y_true = pd.Series(rng.integers(0, 2, 200))
    y_proba = rng.uniform(0, 1, 200)
    curve = compute_calibration_curve(y_true, y_proba, n_bins=5)
    assert {"mean_predicted_probability", "observed_frequency"} <= set(curve.columns)


@pytest.mark.parametrize(
    "model_name,expected_kwarg",
    [
        ("logistic_regression", "class_weight"),
        ("random_forest", "class_weight"),
        ("lightgbm", "class_weight"),
        ("catboost", "auto_class_weights"),
        ("xgboost", "scale_pos_weight"),
    ],
)
def test_build_model_applies_class_weight_strategy_per_family(model_name, expected_kwarg):
    """Regression test: `class_imbalance.strategy: class_weight` in
    configs/model_training.yaml was previously declared but never referenced
    anywhere in training code (dead config). `build_model()` must translate
    it into the right constructor kwarg for each model family."""
    y = pd.Series([0] * 85 + [1] * 15)  # imbalanced, like the real ~15% default rate
    model = build_model(model_name, {}, y=y, class_imbalance_strategy="class_weight")
    assert model.get_params().get(expected_kwarg) is not None


def test_build_model_without_strategy_leaves_class_weight_unset():
    model = build_model("random_forest", {}, y=pd.Series([0, 1]), class_imbalance_strategy=None)
    assert model.get_params().get("class_weight") is None


def test_build_model_smote_strategy_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        build_model("random_forest", {}, y=pd.Series([0, 1]), class_imbalance_strategy="smote")


def test_build_model_unknown_strategy_raises_value_error():
    with pytest.raises(ValueError):
        build_model("random_forest", {}, y=pd.Series([0, 1]), class_imbalance_strategy="not-a-real-strategy")


def test_run_stratified_cv_produces_one_score_per_fold():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(scale=0.1, size=100) > 0).astype(int))

    result = run_stratified_cv(lambda: LogisticRegression(), X, y, n_folds=4, random_state=0)
    assert len(result.fold_roc_auc) == 4
    assert 0.0 <= result.mean_roc_auc <= 1.0
    assert result.oof_predictions is not None
    assert len(result.oof_predictions) == len(X)
