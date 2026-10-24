"""
Model comparison table builder.

Design decision: every candidate (5 base models + stacking + blending) is
scored on the SAME held-out validation split with the SAME metric set, and
the results are collected into one dataframe. This becomes the artifact
behind the "model comparison dashboard" in the Streamlit frontend and is
logged to MLflow as a CSV so the winning-model decision is auditable rather
than a black-box "we picked XGBoost" statement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from core.logging import get_logger

logger = get_logger(__name__)


def evaluate_candidate(name: str, y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (y_proba >= threshold).astype(int)
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
        "brier_score": round(brier_score_loss(y_true, y_proba), 4),
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "f1": round(f1_score(y_true, preds, zero_division=0), 4),
    }


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def select_best_model(comparison_df: pd.DataFrame, metric: str = "pr_auc") -> str:
    best_row = comparison_df.sort_values(metric, ascending=False).iloc[0]
    logger.info(
        "Best model selected",
        extra={"context": {"model": best_row["model"], "metric": metric, "value": best_row[metric]}},
    )
    return str(best_row["model"])
