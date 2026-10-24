"""
Feature drift, target drift, and prediction/confidence drift detection.

Design decision: Population Stability Index (PSI) for numeric features is
the industry-standard metric in credit risk specifically (it's what most
banks' model-risk-management policies already reference), computed via
consistent quantile binning learned on the REFERENCE window so the same bin
edges are reused on the current window — this is what makes PSI meaningful
(comparing two different binning schemes silently invalidates the metric).
Categorical drift uses the same PSI formula over category proportions. The
conventional thresholds (PSI < 0.1 = no drift, 0.1-0.25 = moderate, > 0.25 =
significant) are used to auto-classify severity rather than leaving a raw
number for the caller to interpret.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.logging import get_logger

logger = get_logger(__name__)

PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.25


def _psi_from_proportions(reference_pct: np.ndarray, current_pct: np.ndarray) -> float:
    eps = 1e-4
    reference_pct = np.clip(reference_pct, eps, None)
    current_pct = np.clip(current_pct, eps, None)
    return float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))


def compute_numeric_psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    bin_edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(bin_edges) < 3:  # degenerate (near-constant) feature
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    return _psi_from_proportions(ref_pct, cur_pct)


def compute_categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
    ref_pct = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    cur_pct = current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return _psi_from_proportions(ref_pct, cur_pct)


def _severity(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        return "significant"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "none"


@dataclass
class DriftReport:
    feature_drift: pd.DataFrame
    target_drift_psi: float | None = None
    prediction_drift_psi: float | None = None
    overall_severity: str = "none"


def compute_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for col in numeric_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = compute_numeric_psi(reference_df[col], current_df[col])
        rows.append({"feature": col, "type": "numeric", "psi": round(psi, 4), "severity": _severity(psi)})

    for col in categorical_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = compute_categorical_psi(reference_df[col], current_df[col])
        rows.append({"feature": col, "type": "categorical", "psi": round(psi, 4), "severity": _severity(psi)})

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def run_drift_detection(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    reference_target: pd.Series | None = None,
    current_target: pd.Series | None = None,
    reference_predictions: pd.Series | None = None,
    current_predictions: pd.Series | None = None,
) -> DriftReport:
    feature_drift = compute_feature_drift(reference_df, current_df, numeric_columns, categorical_columns)

    target_psi = None
    if reference_target is not None and current_target is not None:
        target_psi = compute_numeric_psi(reference_target.astype(float), current_target.astype(float), n_bins=2)

    pred_psi = None
    if reference_predictions is not None and current_predictions is not None:
        pred_psi = compute_numeric_psi(reference_predictions, current_predictions)

    severities = list(feature_drift["severity"]) if len(feature_drift) else []
    if target_psi is not None:
        severities.append(_severity(target_psi))
    if pred_psi is not None:
        severities.append(_severity(pred_psi))
    overall = "significant" if "significant" in severities else ("moderate" if "moderate" in severities else "none")

    logger.info("Drift detection complete", extra={"context": {"overall_severity": overall}})
    return DriftReport(
        feature_drift=feature_drift,
        target_drift_psi=target_psi,
        prediction_drift_psi=pred_psi,
        overall_severity=overall,
    )
