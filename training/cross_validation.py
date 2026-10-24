"""
Stratified K-Fold cross-validation utilities.

Design decision: we always stratify on the target (class imbalance is
significant here — ~15% default rate) and expose both per-fold and
aggregate metrics, because a single train/test split can be dangerously
noisy at this default rate (a "lucky" split can swing ROC-AUC by several
points). Reporting fold-level std-dev alongside the mean gives an honest
picture of model stability, which matters more than the point estimate when
deciding between two close models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CVResult:
    fold_roc_auc: list[float] = field(default_factory=list)
    fold_pr_auc: list[float] = field(default_factory=list)
    oof_predictions: np.ndarray | None = None

    @property
    def mean_roc_auc(self) -> float:
        return float(np.mean(self.fold_roc_auc))

    @property
    def std_roc_auc(self) -> float:
        return float(np.std(self.fold_roc_auc))

    @property
    def mean_pr_auc(self) -> float:
        return float(np.mean(self.fold_pr_auc))

    @property
    def std_pr_auc(self) -> float:
        return float(np.std(self.fold_pr_auc))

    def summary(self) -> dict:
        return {
            "roc_auc_mean": round(self.mean_roc_auc, 4),
            "roc_auc_std": round(self.std_roc_auc, 4),
            "pr_auc_mean": round(self.mean_pr_auc, 4),
            "pr_auc_std": round(self.std_pr_auc, 4),
        }


def run_stratified_cv(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
) -> CVResult:
    """Runs Stratified K-Fold CV, returning per-fold metrics and out-of-fold predictions.

    `model_factory` is a zero-arg callable returning a FRESH unfitted
    estimator each call (so folds don't share fitted state).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X))
    result = CVResult()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model = model_factory()
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_proba

        roc = roc_auc_score(y_val, val_proba)
        pr = average_precision_score(y_val, val_proba)
        result.fold_roc_auc.append(roc)
        result.fold_pr_auc.append(pr)

        logger.info(
            f"CV fold {fold_idx + 1}/{n_folds} complete",
            extra={"context": {"roc_auc": round(roc, 4), "pr_auc": round(pr, 4)}},
        )

    result.oof_predictions = oof_preds
    return result
