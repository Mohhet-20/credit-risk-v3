"""
Decision threshold optimization.

Design decision: 0.5 is not a meaningful decision boundary for an imbalanced
credit risk problem — it's a leftover convention from balanced classification.
We support three explicit business-aligned strategies:
  - "f1": maximize F1 (balances precision/recall equally — a reasonable
    generic default when no cost matrix is available).
  - "youden_j": maximize sensitivity + specificity - 1 (the point on the ROC
    curve furthest from the diagonal) — a standard choice in clinical/credit
    scoring literature when false positives and negatives are considered
    roughly symmetric in *rank* importance.
  - "cost_sensitive": directly minimizes expected business cost using the
    cost matrix from `model_config.yaml` (cost_false_negative vastly
    exceeds cost_false_positive for credit risk — approving a defaulter is
    far more expensive than rejecting a creditworthy applicant). This is the
    production default because it's the only one that encodes an actual
    business objective rather than a statistical proxy for one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ThresholdResult:
    threshold: float
    method: str
    metric_value: float


def optimize_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    method: str = "cost_sensitive",
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    n_candidates: int = 200,
) -> ThresholdResult:
    y_true_arr = np.asarray(y_true)
    candidates = np.linspace(0.01, 0.99, n_candidates)

    if method == "f1":
        scores = [f1_score(y_true_arr, (y_proba >= t).astype(int), zero_division=0) for t in candidates]
        best_idx = int(np.argmax(scores))
        return ThresholdResult(
            threshold=float(candidates[best_idx]), method=method, metric_value=float(scores[best_idx])
        )

    if method == "youden_j":
        fpr, tpr, roc_thresholds = roc_curve(y_true_arr, y_proba)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        thr = float(roc_thresholds[best_idx])
        thr = min(max(thr, 0.0), 1.0)
        return ThresholdResult(threshold=thr, method=method, metric_value=float(j_scores[best_idx]))

    if method == "cost_sensitive":
        costs = []
        for t in candidates:
            preds = (y_proba >= t).astype(int)
            fn = int(((preds == 0) & (y_true_arr == 1)).sum())
            fp = int(((preds == 1) & (y_true_arr == 0)).sum())
            total_cost = fn * cost_fn + fp * cost_fp
            costs.append(total_cost)
        best_idx = int(np.argmin(costs))
        return ThresholdResult(
            threshold=float(candidates[best_idx]), method=method, metric_value=float(costs[best_idx])
        )

    raise ValueError(f"Unknown threshold optimization method: {method}")


def evaluate_at_threshold(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import confusion_matrix, precision_score, recall_score

    preds = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "threshold": round(threshold, 4),
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "f1": round(f1_score(y_true, preds, zero_division=0), 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "approval_rate": round(1 - preds.mean(), 4),
    }
