"""
Ensemble construction: stacking and blending on top of the tuned base models.

Design decision:
  - Stacking uses sklearn's `StackingClassifier` with the ALREADY-TUNED base
    estimators (from Optuna) and a simple logistic-regression meta-learner
    trained on out-of-fold base predictions — this is the standard
    leakage-safe stacking recipe (sklearn handles the internal CV
    automatically for the meta-features).
  - Blending is a lighter-weight alternative: base models are trained on a
    "blend train" split, predictions are generated on a held-out "blend
    holdout" split, and a meta-learner is fit on those holdout predictions.
    It's cheaper than stacking (no internal CV) and is included specifically
    to demonstrate the technique distinctly from stacking, as requested — in
    practice we still select whichever ensemble wins on validation PR-AUC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from core.logging import get_logger

logger = get_logger(__name__)

# Supported values for `configs/model_training.yaml`'s `ensemble.stacking_final_estimator` /
# the meta-learner used by `BlendingEnsemble`. A factory (not a hardcoded
# `LogisticRegression()`) so the config key actually controls behavior instead
# of being silently ignored.
META_LEARNER_BUILDERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000),
}


def _build_meta_learner(name: str):
    try:
        return META_LEARNER_BUILDERS[name]()
    except KeyError:
        raise ValueError(
            f"Unsupported ensemble meta-learner {name!r}; supported values: " f"{sorted(META_LEARNER_BUILDERS)}"
        ) from None


def build_stacking_ensemble(
    base_estimators: list[tuple[str, Any]], cv: int = 5, final_estimator_name: str = "logistic_regression"
) -> StackingClassifier:
    """`base_estimators`: list of (name, fitted_or_unfitted_estimator) tuples.
    StackingClassifier refits base estimators internally via CV, so passing
    unfitted clones with the tuned hyperparameters already set is correct.
    """
    meta_learner = _build_meta_learner(final_estimator_name)
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=cv,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


class BlendingEnsemble:
    """Manual blending ensemble: trains base models on a blend-train split,
    fits a meta-learner on their predictions over a held-out blend split."""

    def __init__(
        self,
        base_estimators: list[tuple[str, Any]],
        holdout_fraction: float = 0.2,
        random_state: int = 42,
        meta_learner_name: str = "logistic_regression",
    ):
        self.base_estimators = base_estimators
        self.holdout_fraction = holdout_fraction
        self.random_state = random_state
        self.meta_learner = _build_meta_learner(meta_learner_name)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BlendingEnsemble:
        X_train, X_hold, y_train, y_hold = train_test_split(
            X, y, test_size=self.holdout_fraction, stratify=y, random_state=self.random_state
        )
        self._fitted_base = []
        meta_features = np.zeros((len(X_hold), len(self.base_estimators)))
        for i, (name, estimator) in enumerate(self.base_estimators):
            estimator.fit(X_train, y_train)
            self._fitted_base.append((name, estimator))
            meta_features[:, i] = estimator.predict_proba(X_hold)[:, 1]

        self.meta_learner.fit(meta_features, y_hold)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = np.column_stack([est.predict_proba(X)[:, 1] for _, est in self._fitted_base])
        return self.meta_learner.predict_proba(meta_features)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
