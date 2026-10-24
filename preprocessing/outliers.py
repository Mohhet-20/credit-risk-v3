"""
Outlier detection and handling.

Design decision: we WINSORIZE (clip to learned percentile bounds) rather than
drop outlier rows. In credit risk, extreme-but-real values (very high income,
very high utilization) are often the most informative for separating good
from bad borrowers — dropping them destroys signal and can itself introduce
survivorship bias. Winsorizing bounds are learned on training data only.
An IQR-based flagging utility is also provided for the data-quality report
(counts/percentages of outliers), independent of the clipping transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OutlierWinsorizer(BaseEstimator, TransformerMixin):
    """Clips numeric columns to [lower_quantile, upper_quantile] learned at fit time."""

    columns: list[str] = field(default_factory=list)
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99

    def fit(self, X: pd.DataFrame, y=None) -> OutlierWinsorizer:
        self._bounds: dict[str, tuple[float, float]] = {}
        for col in self.columns:
            if col not in X.columns:
                continue
            lo = X[col].quantile(self.lower_quantile)
            hi = X[col].quantile(self.upper_quantile)
            self._bounds[col] = (lo, hi)
        logger.info("OutlierWinsorizer fit", extra={"context": {"n_columns": len(self._bounds)}})
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self._bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lo, upper=hi)
        return X


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str], k: float = 1.5) -> pd.DataFrame:
    """Returns a per-column summary of IQR-based outlier counts, for data-quality reporting."""
    records = []
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_outliers = int(((series < lo) | (series > hi)).sum())
        records.append(
            {
                "column": col,
                "n_outliers": n_outliers,
                "pct_outliers": round(n_outliers / max(len(series), 1) * 100, 3),
                "lower_bound": round(float(lo), 3),
                "upper_bound": round(float(hi), 3),
            }
        )
    if not records:
        # No numeric columns to check (e.g. an all-categorical extract) --
        # return an empty frame with the expected schema rather than letting
        # `.sort_values("pct_outliers", ...)` raise a KeyError on a
        # column-less empty DataFrame.
        return pd.DataFrame(columns=["column", "n_outliers", "pct_outliers", "lower_bound", "upper_bound"])
    return pd.DataFrame(records).sort_values("pct_outliers", ascending=False)
