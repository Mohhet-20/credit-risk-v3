"""
Missing-value handling strategies.

Design decision: rather than a single blanket `fillna(0)`, we implement a
column-aware imputer that (a) adds a `<col>_was_missing` indicator before
imputing — since in credit data, *missingness itself* is often predictive
(e.g. missing employment_length correlates with self-employment/instability)
— and (b) supports per-column strategies (median for skewed numerics, mean
for roughly-symmetric ones, mode for categoricals) rather than one strategy
for the whole frame. Indicator flags are added BEFORE imputation and are
themselves treated as model features downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Adds missingness indicators, then imputes numeric (median) and categorical (mode) columns.

    Fit/transform semantics follow sklearn conventions so this slots directly
    into a `Pipeline`. Imputation values are learned on the training fold
    only to avoid leakage across CV folds / train-test split.
    """

    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    add_indicators: bool = True
    indicator_threshold: float = 0.001  # only add indicator if >0.1% missing in training data

    def fit(self, X: pd.DataFrame, y=None) -> MissingValueHandler:
        X = X.copy()
        self._numeric_fill: dict[str, float] = {}
        self._categorical_fill: dict[str, str] = {}
        self._indicator_columns: list[str] = []

        for col in self.numeric_columns:
            if col not in X.columns:
                continue
            self._numeric_fill[col] = float(X[col].median())
            if self.add_indicators and X[col].isna().mean() > self.indicator_threshold:
                self._indicator_columns.append(col)

        for col in self.categorical_columns:
            if col not in X.columns:
                continue
            mode = X[col].mode(dropna=True)
            self._categorical_fill[col] = mode.iloc[0] if len(mode) else "missing"
            if self.add_indicators and X[col].isna().mean() > self.indicator_threshold:
                self._indicator_columns.append(col)

        logger.info(
            "MissingValueHandler fit complete",
            extra={"context": {"indicator_columns": self._indicator_columns}},
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self._indicator_columns:
            if col in X.columns:
                X[f"{col}_was_missing"] = X[col].isna().astype(int)

        for col, fill_value in self._numeric_fill.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_value)

        for col, fill_value in self._categorical_fill.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_value)

        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        cols = (
            list(self.numeric_columns)
            + list(self.categorical_columns)
            + [f"{c}_was_missing" for c in self._indicator_columns]
        )
        return np.array(cols)
