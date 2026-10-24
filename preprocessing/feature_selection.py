"""
Feature selection pipeline.

Design decision: we chain three complementary filters rather than relying on
a single method, because each catches a different failure mode:
  1. Variance threshold  -> drops near-constant columns (zero signal, wastes
     model capacity, and confuses SHAP importance rankings with noise).
  2. Pairwise correlation pruning -> drops one of any two features correlated
     above `corr_threshold`, keeping the one more correlated with the
     target. Reduces multicollinearity, which matters most for the logistic
     regression baseline and for making SHAP dependence plots interpretable
     (two 0.95-correlated features "split" importance in confusing ways).
  3. Model-based importance ranking (Random Forest permutation importance) ->
     drops the bottom-k features that add ~nothing to a strong model, which
     also directly reduces production inference latency and monitoring
     surface area (fewer features to drift-check).
All thresholds are learned on training data only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureSelector:
    variance_threshold: float = 0.0
    corr_threshold: float = 0.95
    importance_top_k: int | None = None  # None = keep all that survive filters
    random_state: int = 42

    selected_features_: list[str] = field(default_factory=list, init=False)
    dropped_low_variance_: list[str] = field(default_factory=list, init=False)
    dropped_correlated_: list[str] = field(default_factory=list, init=False)
    importance_table_: pd.DataFrame | None = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FeatureSelector:
        # `include=[np.number, "bool"]` -- one-hot encoded columns upstream
        # (`OneHotFrameEncoder`, via `pd.get_dummies`) are `bool` dtype, which
        # `np.number` alone excludes. Without `"bool"` here, every dummy
        # column silently skips variance/correlation/importance pruning
        # entirely regardless of `corr_threshold` or redundancy.
        numeric_X = X.select_dtypes(include=[np.number, "bool"]).fillna(0)

        # 1. Variance threshold
        vt = VarianceThreshold(threshold=self.variance_threshold)
        vt.fit(numeric_X)
        kept_mask = vt.get_support()
        self.dropped_low_variance_ = [c for c, k in zip(numeric_X.columns, kept_mask, strict=False) if not k]
        survivors = [c for c, k in zip(numeric_X.columns, kept_mask, strict=False) if k]

        # 2. Correlation pruning (keep the twin more correlated with target)
        corr_matrix = numeric_X[survivors].corr().abs()
        target_corr = numeric_X[survivors].apply(lambda col: abs(np.corrcoef(col, y)[0, 1]))
        to_drop: set[str] = set()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        for col in upper.columns:
            for row in upper.index:
                val = upper.loc[row, col]
                if pd.notna(val) and val > self.corr_threshold:
                    weaker = row if target_corr.get(row, 0) < target_corr.get(col, 0) else col
                    to_drop.add(weaker)
        self.dropped_correlated_ = sorted(to_drop)
        survivors = [c for c in survivors if c not in to_drop]

        # 3. Model-based permutation importance ranking
        if len(survivors) > 0:
            rf = RandomForestClassifier(n_estimators=150, max_depth=8, n_jobs=-1, random_state=self.random_state)
            rf.fit(numeric_X[survivors], y)
            result = permutation_importance(
                rf, numeric_X[survivors], y, n_repeats=3, random_state=self.random_state, n_jobs=-1
            )
            self.importance_table_ = (
                pd.DataFrame(
                    {
                        "feature": survivors,
                        "importance_mean": result.importances_mean,
                        "importance_std": result.importances_std,
                    }
                )
                .sort_values("importance_mean", ascending=False)
                .reset_index(drop=True)
            )

            if self.importance_top_k is not None:
                survivors = self.importance_table_.head(self.importance_top_k)["feature"].tolist()

        # Always retain non-numeric (categorical) columns untouched — selection
        # here only prunes the numeric feature space.
        categorical_cols = [c for c in X.columns if c not in numeric_X.columns]
        self.selected_features_ = survivors + categorical_cols

        logger.info(
            "FeatureSelector fit complete",
            extra={
                "context": {
                    "n_input": X.shape[1],
                    "n_selected": len(self.selected_features_),
                    "dropped_low_variance": len(self.dropped_low_variance_),
                    "dropped_correlated": len(self.dropped_correlated_),
                }
            },
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.selected_features_ if c in X.columns]
        return X[cols]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
