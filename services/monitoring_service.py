"""
Monitoring business logic: prediction distribution, latency percentiles, and
PSI-based data/feature drift against the training reference window.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from configs.settings import get_settings
from core.cache import get_cache
from core.constants import RAW_CATEGORICAL_FEATURES, RAW_NUMERIC_FEATURES
from data_quality.drift_detection import run_drift_detection
from repositories.prediction_repository import PredictionRepository

STATS_CACHE_TTL_SECONDS = 15.0


class MonitoringService:
    def __init__(self, prediction_repo: PredictionRepository):
        self.prediction_repo = prediction_repo

    def get_stats(self, days: int = 30) -> dict:
        """Short-TTL cached: the executive/monitoring dashboards poll this
        endpoint frequently, but a 15-second staleness window is invisible
        to a human looking at a dashboard and saves a DB aggregation query
        per poll. No explicit invalidation hook is needed (unlike the model
        comparison cache) -- new predictions arrive continuously, so there's
        no single write event to hang a "bust now" call off of; a short TTL
        is the natural fit."""
        cache = get_cache()
        key = f"monitoring:stats:{days}"
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = self.prediction_repo.stats_last_n_days(days=days)
        cache.set(key, result, STATS_CACHE_TTL_SECONDS)
        return result

    def get_distribution(self, days: int = 30) -> dict:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        predictions = self.prediction_repo.list_since(since=since)
        if not predictions:
            return {"n": 0, "histogram": [], "risk_band_counts": {}, "decision_counts": {}}

        probas = [p.predicted_probability for p in predictions]
        counts, edges = np.histogram(probas, bins=10, range=(0, 1))
        risk_bands: dict[str, int] = {}
        decisions: dict[str, int] = {}
        for p in predictions:
            risk_bands[p.risk_band] = risk_bands.get(p.risk_band, 0) + 1
            decisions[p.decision] = decisions.get(p.decision, 0) + 1

        return {
            "n": len(predictions),
            "histogram": [
                {
                    "bin_start": round(float(edges[i]), 2),
                    "bin_end": round(float(edges[i + 1]), 2),
                    "count": int(counts[i]),
                }
                for i in range(len(counts))
            ],
            "risk_band_counts": risk_bands,
            "decision_counts": decisions,
        }

    def get_drift(self) -> dict:
        """Compares the original training data (reference window) against the
        most recently scored applications (current window) recorded in the DB."""
        settings = get_settings()
        reference_path = settings.data_raw_dir / "credit_applications.csv"
        if not reference_path.exists():
            return {"error": "No reference dataset found; drift cannot be computed."}

        reference_df = pd.read_csv(reference_path)
        recent = self.prediction_repo.list_predictions(limit=2000)
        if len(recent) < 30:
            return {"error": "Not enough recent predictions to compute meaningful drift (need >= 30)."}

        current_df = pd.DataFrame([p.input_features for p in recent])
        numeric_cols = [c for c in RAW_NUMERIC_FEATURES if c in current_df.columns]
        categorical_cols = [c for c in RAW_CATEGORICAL_FEATURES if c in current_df.columns]

        report = run_drift_detection(reference_df, current_df, numeric_cols, categorical_cols)
        return {
            "overall_severity": report.overall_severity,
            "feature_drift": report.feature_drift.to_dict(orient="records"),
        }

    def get_latency_stats(self) -> dict:
        predictions = self.prediction_repo.list_predictions(limit=1000)
        latencies = [p.latency_ms for p in predictions if p.latency_ms is not None]
        if not latencies:
            return {"n": 0}
        arr = np.array(latencies)
        return {
            "n": len(arr),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "max_ms": round(float(arr.max()), 2),
        }
