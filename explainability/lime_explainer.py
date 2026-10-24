"""
LIME (Local Interpretable Model-agnostic Explanations) as an optional
comparison method alongside the platform's primary SHAP explainer.

Design decision: SHAP and LIME answer "why did the model predict this?"
via genuinely different mechanisms -- SHAP computes exact/approximate
Shapley values from cooperative game theory; LIME perturbs samples around
the instance and fits a local weighted *linear* surrogate model. On tabular
credit data with real feature interactions (which the champion tree models
exploit and SHAP's TreeExplainer captures exactly), LIME's linear surrogate
can disagree with SHAP -- that disagreement is informative for a risk
reviewer, not a bug to reconcile. This module is kept as a genuinely
optional add-on: `/explain` (SHAP) remains the platform's primary
explanation endpoint and is completely unchanged by this addition.

Design decision: a fresh `LimeTabularExplainer` is built per champion model
version (cached alongside the existing SHAP background-sample cache in
`api/routers/explain.py`), not per request -- fitting LIME's internal
discretizer over the background sample is the expensive part, and it only
needs to change when the champion model (and therefore the reference
population) changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LimeExplainer:
    background: pd.DataFrame
    feature_names: list[str]
    class_names: tuple[str, str] = ("repaid", "default")
    num_samples: int = 500

    _explainer: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._explainer = LimeTabularExplainer(
            training_data=self.background.to_numpy(dtype=float),
            feature_names=self.feature_names,
            class_names=list(self.class_names),
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

    def explain_row(
        self,
        predict_proba_positive_fn: Callable[[pd.DataFrame], np.ndarray],
        x_row: pd.DataFrame,
        num_features: int = 10,
    ) -> dict:
        """`predict_proba_positive_fn` returns P(default) for each row of a
        DataFrame -- LIME's API wants a two-column [P(negative), P(positive)]
        array, which we synthesize here so callers only need the same
        single-column predict function already used by SHAP/counterfactuals."""
        instance = x_row.iloc[0].to_numpy(dtype=float)

        def _predict_fn(data: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(data, columns=self.feature_names)
            proba_positive = np.asarray(predict_proba_positive_fn(df))
            return np.column_stack([1 - proba_positive, proba_positive])

        explanation = self._explainer.explain_instance(
            instance,
            _predict_fn,
            num_features=num_features,
            num_samples=self.num_samples,
            labels=(1,),
        )
        contributions = [{"rule": rule, "weight": float(weight)} for rule, weight in explanation.as_list(label=1)]
        local_pred = None
        if hasattr(explanation, "local_pred") and explanation.local_pred is not None:
            local_pred = float(np.ravel(explanation.local_pred)[0])

        return {
            "intercept": float(explanation.intercept[1]),
            "local_prediction": local_pred,
            "contributions": contributions,
        }
