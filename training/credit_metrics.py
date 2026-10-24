"""
Credit-risk validation metrics: the numbers a model-risk reviewer asks for.

ROC-AUC/PR-AUC (see `training/model_comparison.py`) are generic ML metrics.
Credit model validation (SR 11-7 / OCC 2011-12 style) additionally expects:
  - Gini (= 2*AUC - 1): the industry's headline discrimination number.
  - KS statistic: max separation between the cumulative good and bad
    score distributions -- the classic scorecard acceptance metric.
  - A decile (gains/lift) table: bad rate, cumulative capture and lift by
    score decile, which is how a credit committee actually reads a model.
  - Bootstrap confidence intervals, because a point AUC on a few thousand
    loans is not a finding without its uncertainty band.
  - Hosmer-Lemeshow-style calibration-in-the-large by decile (predicted vs
    observed default rate), since PD feeds expected-loss and capital.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def gini(y_true, y_proba) -> float:
    return float(2 * roc_auc_score(y_true, y_proba) - 1)


def ks_statistic(y_true, y_proba) -> float:
    """Max vertical gap between the bad and good cumulative score
    distributions, i.e. max(TPR - FPR) over all cut-offs. Computed from the
    ROC curve rather than `scipy.stats.ks_2samp`, which also computes an
    exact p-value -- prohibitively slow at portfolio sizes and inside a
    bootstrap, and not needed here."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def decile_table(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame:
    """Gains/lift table, riskiest decile first. `capture_rate` is the share
    of all defaults caught by rejecting this decile and every riskier one."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_proba)})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), n_bins, labels=range(1, n_bins + 1))
    overall_bad_rate = df["y"].mean()
    out = (
        df.groupby("decile", observed=True)
        .agg(n=("y", "size"), defaults=("y", "sum"), min_pd=("p", "min"), max_pd=("p", "max"), mean_pd=("p", "mean"))
        .reset_index()
    )
    out["bad_rate"] = out["defaults"] / out["n"]
    out["capture_rate"] = out["defaults"].cumsum() / max(out["defaults"].sum(), 1)
    out["lift"] = out["bad_rate"] / overall_bad_rate if overall_bad_rate > 0 else np.nan
    out["decile"] = out["decile"].astype(int)
    return out.round(4)


def bootstrap_ci(
    y_true, y_proba, metric, n_boot: int = 500, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        stats.append(metric(y_true[idx], y_proba[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def credit_validation_metrics(y_true, y_proba, *, with_ci: bool = True) -> dict:
    auc = roc_auc_score(y_true, y_proba)
    out = {
        "roc_auc": round(float(auc), 4),
        "gini": round(gini(y_true, y_proba), 4),
        "ks": round(ks_statistic(y_true, y_proba), 4),
        "observed_default_rate": round(float(np.mean(y_true)), 4),
        "mean_predicted_pd": round(float(np.mean(y_proba)), 4),
        "n": int(len(y_true)),
        "n_defaults": int(np.sum(y_true)),
    }
    if with_ci:
        lo, hi = bootstrap_ci(y_true, y_proba, roc_auc_score)
        out["roc_auc_ci95"] = [round(lo, 4), round(hi, 4)]
        lo, hi = bootstrap_ci(y_true, y_proba, ks_statistic)
        out["ks_ci95"] = [round(lo, 4), round(hi, 4)]
    return out
