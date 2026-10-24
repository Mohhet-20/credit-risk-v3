"""
K-Fold smoothed target encoding for high-cardinality categorical features.

Split out into its own module (rather than living in `encoders.py`) purely
for file-organization clarity as the preprocessing package grew.

Design decision: K-Fold target encoding (rather than plain one-hot) is used
for high-cardinality columns like `state` because one-hot would add dozens of
sparse columns with weak individual signal. The *out-of-fold* variant is
essential: computing a naive global mean-per-category on the full training
set and using it as a feature would leak the label into the training rows
that helped compute their own encoding. We instead compute each row's
encoding using only the OTHER folds, and separately fit a full-data map
(via `.fit()` / `KFoldTargetEncoder`) for use at inference time, when there's
no leakage concern because the "current" row was never used to build the map.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Out-of-fold smoothed target encoding for high-cardinality categoricals."""

    columns: list[str] = field(default_factory=list)
    n_folds: int = 5
    smoothing: float = 10.0
    random_state: int = 42

    def fit(self, X: pd.DataFrame, y: pd.Series) -> KFoldTargetEncoder:
        y = pd.Series(y).reset_index(drop=True)
        self._global_mean = float(y.mean())
        self._category_maps: dict[str, dict] = {}
        for col in self.columns:
            if col not in X.columns:
                continue
            stats = pd.DataFrame({col: X[col].reset_index(drop=True), "y": y}).groupby(col)["y"].agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + self._global_mean * self.smoothing) / (
                stats["count"] + self.smoothing
            )
            self._category_maps[col] = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self._category_maps.items():
            if col in X.columns:
                X[f"{col}_target_enc"] = X[col].map(mapping).fillna(self._global_mean)
                X = X.drop(columns=[col])
        return X

    def fit_transform_oof(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Out-of-fold transform for TRAINING data only — call `.transform()` for inference/val/test."""
        self.fit(X, y)
        return compute_oof_target_encoding(
            X,
            y,
            columns=self.columns,
            n_folds=self.n_folds,
            smoothing=self.smoothing,
            random_state=self.random_state,
        )


def compute_oof_target_encoding(
    X: pd.DataFrame,
    y: pd.Series,
    columns: list[str],
    n_folds: int = 5,
    smoothing: float = 10.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Pure function: returns an out-of-fold target-encoded copy of X."""
    y = pd.Series(y).reset_index(drop=True)
    X_in = X.reset_index(drop=True).copy()
    global_mean = float(y.mean())
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    X_out = X_in.copy()
    for col in columns:
        if col not in X_in.columns:
            continue
        oof_values = np.full(len(X_in), np.nan)
        for train_idx, holdout_idx in skf.split(X_in, y):
            stats = (
                pd.DataFrame({col: X_in.loc[train_idx, col], "y": y.loc[train_idx]})
                .groupby(col)["y"]
                .agg(["mean", "count"])
            )
            smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            mapped = X_in.loc[holdout_idx, col].map(smoothed).fillna(global_mean)
            oof_values[holdout_idx] = mapped.values
        X_out[f"{col}_target_enc"] = oof_values
        X_out = X_out.drop(columns=[col])

    return X_out


class OneHotFrameEncoder(BaseEstimator, TransformerMixin):
    """Thin pandas-native one-hot encoder that remembers training-time columns
    so validation/inference frames always align (missing dummy columns are
    added as zeros; unseen categories are silently dropped)."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y=None) -> OneHotFrameEncoder:
        dummies = pd.get_dummies(X[self.columns], prefix=self.columns, dummy_na=False)
        self._output_columns = dummies.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dummies = pd.get_dummies(X[self.columns], prefix=self.columns, dummy_na=False)
        dummies = dummies.reindex(columns=self._output_columns, fill_value=0)
        X = X.drop(columns=self.columns).reset_index(drop=True)
        return pd.concat([X, dummies.reset_index(drop=True)], axis=1)
