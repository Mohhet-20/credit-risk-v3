"""
SHAP-based explainability: global importance, local (per-applicant)
explanations, dependence/interaction structure, and decision plots.

Design decision: the champion model can be a plain tree ensemble OR a
`StackingClassifier` / `CalibratedClassifierCV` meta-estimator, and SHAP's
fast `TreeExplainer` only works on the former. Rather than forcing a single
explainer type, `ModelExplainer` picks the fastest applicable SHAP backend:
`TreeExplainer` when the underlying estimator is a recognized tree model,
otherwise falls back to `shap.Explainer` wrapping `predict_proba`
(permutation-based), which works for any black-box estimator at the cost of
speed — so we always explain on a bounded sample (`max_background` /
`max_explain`) rather than the full dataset, which is standard practice for
SHAP in production (explaining every row of a large batch is rarely
necessary or affordable).

Note: the feature pipeline's one-hot encoding step produces boolean dtype
columns (pandas' `get_dummies` default). A DataFrame mixing bool/int/float
columns can coerce to an `object` numpy array on `.values`/`.to_numpy()` in
some pandas versions, which breaks SHAP's internal `isclose`/`isfinite`
casting, and forces `.iloc[0]` (a single row) into an object-dtype Series
whose `.tolist()` retains numpy scalar types instead of native Python types
(not JSON-serializable). We defensively cast/unwrap in both directions here
rather than relying on upstream dtype discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import shap

from core.logging import get_logger

logger = get_logger(__name__)

_TREE_MODEL_MARKERS = (
    "RandomForestClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "CatBoostClassifier",
    "GradientBoostingClassifier",
    "ExtraTreesClassifier",
)


def _to_float_frame(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype(np.float64)


def _unwrap_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


@dataclass
class ModelExplainer:
    model: Any
    feature_names: list[str]
    max_background: int = 100
    max_explain: int = 200

    _explainer: Any = field(default=None, init=False)
    _backend: str = field(default="", init=False)
    _is_tree: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._is_tree = type(self._innermost_estimator()).__name__ in _TREE_MODEL_MARKERS

    def _innermost_estimator(self) -> Any:
        """Unwraps CalibratedClassifierCV / PrefitCalibratedModel to get at the
        underlying tree model, if applicable. Calibration is a monotone map of
        the raw score, so attributions on the inner model rank features the
        same way they rank for the calibrated PD."""
        model = self.model
        if hasattr(model, "fitted_model"):
            return model.fitted_model
        if hasattr(model, "calibrated_classifiers_"):
            try:
                return model.calibrated_classifiers_[0].estimator
            except Exception:  # noqa: BLE001
                return model
        if hasattr(model, "estimator"):
            return model.estimator
        return model

    def build(self, X_background: pd.DataFrame) -> ModelExplainer:
        bg = _to_float_frame(X_background.sample(min(len(X_background), self.max_background), random_state=42))

        if self._is_tree:
            try:
                self._explainer = shap.TreeExplainer(
                    self._innermost_estimator(), data=bg, feature_perturbation="interventional"
                )
                self._backend = "tree"
                logger.info("Using SHAP TreeExplainer")
                return self
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"TreeExplainer unavailable ({exc}), falling back to model-agnostic explainer.")

        feature_names = self.feature_names

        def predict_fn(X: np.ndarray) -> np.ndarray:
            return self.model.predict_proba(pd.DataFrame(X, columns=feature_names))[:, 1]

        self._explainer = shap.Explainer(predict_fn, bg, feature_names=feature_names)
        self._backend = "agnostic"
        logger.info("Using model-agnostic SHAP Explainer (permutation-based)")
        return self

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        if self._explainer is None:
            raise RuntimeError("Call .build(background) before .explain().")
        X_sample = _to_float_frame(X.head(self.max_explain))
        explanation = self._explainer(X_sample)
        if hasattr(explanation, "values") and explanation.values.ndim == 3:
            explanation = explanation[:, :, 1]
        return explanation

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        explanation = self.explain(X)
        mean_abs = np.abs(explanation.values).mean(axis=0)
        df = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def local_explanation(self, X_row: pd.DataFrame) -> dict:
        """Returns a JSON-serializable breakdown for ONE applicant row."""
        explanation = self.explain(X_row.head(1))
        values = explanation.values[0]
        base_value = float(np.ravel(explanation.base_values)[0])
        raw_row_values = [_unwrap_scalar(v) for v in X_row.iloc[0].tolist()]
        contributions = sorted(
            zip(self.feature_names, values, raw_row_values, strict=False),
            key=lambda t: abs(t[1]),
            reverse=True,
        )
        return {
            "base_value": base_value,
            "predicted_value": base_value + float(values.sum()),
            "contributions": [
                {"feature": f, "shap_value": float(v), "feature_value": fv} for f, v, fv in contributions
            ],
        }

    def interaction_values(self, X: pd.DataFrame) -> np.ndarray | None:
        """Only supported for TreeExplainer backends; returns None otherwise."""
        if self._backend != "tree":
            logger.warning("Interaction values require a TreeExplainer backend; skipping.")
            return None
        X_sample = _to_float_frame(X.head(min(len(X), 50)))
        return self._explainer.shap_interaction_values(X_sample)
