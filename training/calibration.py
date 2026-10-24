"""
Probability calibration.

Design decision: raw predict_proba() outputs from tree ensembles (especially
gradient-boosted trees) are notoriously poorly calibrated — they tend to push
probabilities toward the extremes. For a credit risk model, the predicted
probability of default (PD) is not just used for ranking (where calibration
wouldn't matter) but is fed directly into expected-loss calculations
(EL = PD x LGD x EAD) and regulatory capital estimates, where the ACTUAL
probability value matters. We support both:
  - Platt scaling (sigmoid): a single logistic regression on the scores —
    robust with small calibration sets, assumes a sigmoid-shaped miscalibration.
  - Isotonic regression: a non-parametric, monotonic fit — more flexible,
    preferred with more data (we default to isotonic per `model_config.yaml`
    given our calibration fold sizes are in the thousands).
Both are fit via cross-validated `CalibratedClassifierCV` so the calibration
map itself doesn't leak information from the fold it's evaluated on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from core.logging import get_logger

logger = get_logger(__name__)


def calibrate_model(fitted_model, X_calib: pd.DataFrame, y_calib: pd.Series, method: str = "isotonic", cv: int = 5):
    """Wraps an already-fitted estimator with cross-validated calibration.

    `method`: "isotonic" or "sigmoid" (Platt scaling).
    Uses `cv="prefit"`-free CalibratedClassifierCV (cv=int) so an internal
    CV loop refits base estimators on folds of X_calib/y_calib — this uses
    a dedicated calibration split from the caller, not the original train fold.

    Only suitable for plain sklearn-conformant estimators (clonable, expose
    `classes_`/`get_params`) -- stacking/blending ensembles should use
    `calibrate_prefit_model()` instead (see its docstring for why).
    """
    calibrated = CalibratedClassifierCV(fitted_model, method=method, cv=cv)
    calibrated.fit(X_calib, y_calib)
    return calibrated


class PrefitCalibratedModel:
    """Wraps an already-fitted model with a calibration map fit once on a
    held-out calibration set, without requiring the wrapped model to be
    sklearn-clonable or expose `classes_`/`get_params` (needed for
    `training.ensembles.BlendingEnsemble`, a hand-rolled class, and to avoid
    the cost of refitting an expensive stacking/blending ensemble multiple
    times via `CalibratedClassifierCV`'s internal cross-validation). Exposes
    just the `predict_proba`/`predict` surface every downstream consumer
    (scoring, model comparison, the model registry) actually relies on."""

    def __init__(self, fitted_model, calibrator, method: str):
        self.fitted_model = fitted_model
        self.calibrator = calibrator
        self.method = method

    def predict_proba(self, X) -> np.ndarray:
        raw_positive = self.fitted_model.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            calibrated_positive = self.calibrator.predict(raw_positive)
        else:  # sigmoid / Platt scaling
            calibrated_positive = self.calibrator.predict_proba(raw_positive.reshape(-1, 1))[:, 1]
        calibrated_positive = np.clip(calibrated_positive, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated_positive, calibrated_positive])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def calibrate_prefit_model(
    fitted_model, X_calib: pd.DataFrame, y_calib: pd.Series, method: str = "isotonic"
) -> PrefitCalibratedModel:
    """Calibrates an already-fitted model by fitting a calibration map on top
    of its raw `predict_proba` output over a held-out calibration set -- a
    single fit, not `CalibratedClassifierCV`'s cross-validated refit-on-clones
    approach used by `calibrate_model()`. Use this for stacking/blending
    ensembles: they're expensive to refit repeatedly, and `BlendingEnsemble`
    in particular isn't a clonable sklearn estimator at all, so
    `CalibratedClassifierCV` cannot wrap it directly."""
    raw_positive = fitted_model.predict_proba(X_calib)[:, 1]
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_positive, y_calib)
    elif method == "sigmoid":
        calibrator = LogisticRegression()
        calibrator.fit(raw_positive.reshape(-1, 1), y_calib)
    else:
        raise ValueError(f"Unknown calibration method: {method!r}")
    return PrefitCalibratedModel(fitted_model, calibrator, method)


def compute_calibration_curve(y_true: pd.Series, y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Returns a dataframe of (mean predicted prob, observed frequency) per bin
    for plotting reliability diagrams in the explainability / monitoring UI."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted_probability": prob_pred, "observed_frequency": prob_true})


def calibration_metrics(y_true: pd.Series, y_proba: np.ndarray) -> dict:
    """Brier score (lower is better) plus expected calibration error (ECE)."""
    brier = brier_score_loss(y_true, y_proba)
    ece = _expected_calibration_error(y_true.to_numpy(), y_proba)
    return {"brier_score": round(float(brier), 5), "expected_calibration_error": round(float(ece), 5)}


def _expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges[1:-1])
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        bin_confidence = y_proba[mask].mean()
        bin_accuracy = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_confidence - bin_accuracy)
    return ece
