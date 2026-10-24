"""
Application-level anomaly detection.

Design decision: `IsolationForest` operates on the same transformed feature
space as the segmentation module and is fit fresh per request against a
reference sample of the training population. This trades a few milliseconds
of latency for always reflecting the CURRENT champion model's feature space
-- a persisted/cached detector would need explicit invalidation on every
model promotion, which is unnecessary complexity at this project's traffic
scale. A high-QPS production deployment would instead persist a fitted
`IsolationForest` alongside the model artifact in the registry and reload it
on promotion, exactly like `ModelBundle` itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DEFAULT_CONTAMINATION = 0.05


@dataclass
class AnomalyResult:
    is_anomaly: bool
    anomaly_score: float  # higher = more anomalous, clipped to [0, 1]
    top_deviant_features: list[dict]


@dataclass
class AnomalyDetector:
    contamination: float = DEFAULT_CONTAMINATION
    random_state: int = 42
    _model: IsolationForest | None = field(default=None, repr=False)
    _numeric_columns: list[str] = field(default_factory=list, repr=False)
    _reference_mean: pd.Series | None = field(default=None, repr=False)
    _reference_std: pd.Series | None = field(default=None, repr=False)

    def fit(self, transformed: pd.DataFrame) -> AnomalyDetector:
        self._numeric_columns = transformed.select_dtypes(include="number").columns.tolist()
        X = transformed[self._numeric_columns]
        self._reference_mean = X.mean()
        self._reference_std = X.std().replace(0, 1.0)
        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=150,
        ).fit(X)
        return self

    def score(self, transformed_row: pd.DataFrame) -> AnomalyResult:
        if self._model is None:
            raise RuntimeError("AnomalyDetector must be fit before calling score().")
        X = transformed_row[self._numeric_columns]
        raw_score = float(self._model.decision_function(X)[0])  # higher = more normal
        is_anomaly = bool(self._model.predict(X)[0] == -1)
        # decision_function is roughly in [-0.5, 0.5]; rescale to an
        # intuitive [0, 1] "anomalousness" score for the API/frontend.
        anomaly_score = float(np.clip(0.5 - raw_score, 0.0, 1.0))

        z_scores = ((X.iloc[0] - self._reference_mean) / self._reference_std).abs().sort_values(ascending=False)
        top = [{"feature": feat, "z_score": float(z)} for feat, z in z_scores.head(5).items()]
        return AnomalyResult(is_anomaly=is_anomaly, anomaly_score=anomaly_score, top_deviant_features=top)
