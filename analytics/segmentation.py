"""
Customer risk segmentation.

Design decision: segmentation runs on the SAME transformed feature space the
credit-risk model consumes (via `FeaturePipeline.transform`) rather than a
separately engineered feature set. This keeps segment boundaries consistent
with what actually drives the risk score, and lets a single reference sample
(drawn once per request from the training population) back both this module
and the anomaly detector.

`SEGMENT_LABELS` are simple, stably-interpretable heuristics over cluster
centroids (credit_score, revolving_utilization, debt_to_income_ratio) rather
than opaque "Cluster 0/1/2/3" IDs, so a risk analyst gets an actionable
business label out of the box instead of having to reverse-engineer what
each cluster means.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_N_SEGMENTS = 4


@dataclass
class SegmentProfile:
    segment_id: int
    label: str
    size: int
    avg_credit_score: float
    avg_utilization: float
    avg_debt_to_income: float


@dataclass
class CustomerSegmenter:
    n_segments: int = DEFAULT_N_SEGMENTS
    random_state: int = 42
    _scaler: StandardScaler = field(default_factory=StandardScaler, repr=False)
    _kmeans: KMeans | None = field(default=None, repr=False)
    _numeric_columns: list[str] = field(default_factory=list, repr=False)
    profiles_: dict[int, SegmentProfile] = field(default_factory=dict, repr=False)

    def fit(self, transformed: pd.DataFrame, raw: pd.DataFrame) -> CustomerSegmenter:
        self._numeric_columns = transformed.select_dtypes(include="number").columns.tolist()
        X = self._scaler.fit_transform(transformed[self._numeric_columns])
        self._kmeans = KMeans(n_clusters=self.n_segments, random_state=self.random_state, n_init=10).fit(X)
        self.profiles_ = self._build_profiles(raw.reset_index(drop=True), self._kmeans.labels_)
        return self

    def _build_profiles(self, raw: pd.DataFrame, labels: np.ndarray) -> dict[int, SegmentProfile]:
        profiles: dict[int, SegmentProfile] = {}
        for seg_id in sorted(set(labels.tolist())):
            mask = labels == seg_id
            subset = raw.loc[mask]
            avg_score = float(subset["credit_score"].mean())
            avg_util = float(subset["revolving_utilization"].mean())
            avg_dti = float(subset["debt_to_income_ratio"].mean())
            profiles[int(seg_id)] = SegmentProfile(
                segment_id=int(seg_id),
                label=self._label_for(avg_score, avg_util, avg_dti),
                size=int(mask.sum()),
                avg_credit_score=avg_score,
                avg_utilization=avg_util,
                avg_debt_to_income=avg_dti,
            )
        return profiles

    @staticmethod
    def _label_for(avg_score: float, avg_util: float, avg_dti: float) -> str:
        if avg_score >= 720 and avg_util < 0.3:
            return "Prime -- Low Utilization"
        if avg_score >= 720:
            return "Prime -- High Utilization"
        if avg_score >= 650 and avg_dti < 35:
            return "Near-Prime -- Stable"
        if avg_score < 600:
            return "Subprime -- Elevated Risk"
        return "Near-Prime -- Watch List"

    def predict(self, transformed_row: pd.DataFrame) -> SegmentProfile:
        if self._kmeans is None:
            raise RuntimeError("CustomerSegmenter must be fit before calling predict().")
        X = self._scaler.transform(transformed_row[self._numeric_columns])
        seg_id = int(self._kmeans.predict(X)[0])
        return self.profiles_[seg_id]
