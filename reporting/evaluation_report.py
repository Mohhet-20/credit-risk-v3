"""
Evaluation Report: comprehensive metrics for the current champion model,
computed fresh against the labeled reference dataset
(`data/raw/credit_applications.csv`).

Design decision: `training/model_comparison.py::evaluate_candidate` computes
a LEAN metric set (roc_auc, pr_auc, brier, precision/recall/f1 at a 0.5
cutoff) during the training pipeline's cross-model ranking step --
appropriate there, where the goal is fast comparison, not a full report.
This module computes the FULLER set the spec calls for (accuracy, confusion
matrix, calibration curve, lift chart, gain chart, a full threshold sweep)
at report-generation time, evaluated at the model's OWN cost-sensitive
decision threshold (`bundle.metadata["decision_threshold"]`) rather than the
generic 0.5 cutoff used for training-time ranking -- consistent with this
platform's threshold-optimization design (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from configs.settings import get_settings
from core.constants import TARGET_COLUMN
from models.registry import ModelBundle
from reporting.html_report import df_to_html_table, fig_to_base64_img, kpi_html, kpi_row_html, render_report

MAX_EVAL_ROWS = 5000


def _reference_dataset() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_csv(settings.data_raw_dir / "credit_applications.csv")


def compute_evaluation_metrics(bundle: ModelBundle, max_rows: int = MAX_EVAL_ROWS) -> dict:
    df = _reference_dataset()
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)

    y_true = df[TARGET_COLUMN].to_numpy()
    X = df.drop(columns=[TARGET_COLUMN])
    Xt = bundle.feature_pipeline.transform(X)
    y_proba = bundle.model.predict_proba(Xt)[:, 1]

    threshold = float(bundle.metadata.get("decision_threshold", 0.5))
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")

    # Lift / gain: sort by predicted risk descending, bucket into deciles.
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    n = len(y_sorted)
    total_positives = int(y_sorted.sum())
    decile_size = max(1, n // 10)

    gains, lifts = [], []
    cumulative_positives = 0
    for i in range(10):
        end = (i + 1) * decile_size if i < 9 else n
        cumulative_positives += int(y_sorted[i * decile_size : end].sum())
        pct_population = end / n
        pct_positives = cumulative_positives / total_positives if total_positives else 0.0
        gains.append(
            {
                "decile": i + 1,
                "pct_population": round(pct_population, 3),
                "pct_positives_captured": round(pct_positives, 3),
            }
        )
        lifts.append(
            {
                "decile": i + 1,
                "lift": round(pct_positives / pct_population, 3) if pct_population else 0.0,
            }
        )

    threshold_rows = []
    for t in np.linspace(0.05, 0.95, 19):
        preds_t = (y_proba >= t).astype(int)
        threshold_rows.append(
            {
                "threshold": round(float(t), 2),
                "precision": round(float(precision_score(y_true, preds_t, zero_division=0)), 3),
                "recall": round(float(recall_score(y_true, preds_t, zero_division=0)), 3),
                "f1": round(float(f1_score(y_true, preds_t, zero_division=0)), 3),
                "approval_rate": round(float(1 - preds_t.mean()), 3),
            }
        )

    return {
        "n_samples": int(n),
        "decision_threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": cm.tolist(),
        "calibration_curve": {
            "mean_predicted": [round(float(v), 4) for v in mean_pred],
            "fraction_positive": [round(float(v), 4) for v in frac_pos],
        },
        "lift_chart": lifts,
        "gain_chart": gains,
        "threshold_analysis": threshold_rows,
    }


def _confusion_matrix_fig(cm: list[list[int]]):
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.imshow(cm_arr, cmap="Blues")
    labels = ["repaid (0)", "default (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm_arr[i, j]),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black",
            )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def _calibration_fig(calibration: dict):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.plot(calibration["mean_predicted"], calibration["fraction_positive"], marker="o", color="#2c7fb8", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration Curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _lift_gain_fig(gains: list[dict], lifts: list[dict]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    deciles = [g["decile"] for g in gains]

    ax1.plot(deciles, [g["pct_positives_captured"] for g in gains], marker="o", color="#2c7fb8", label="Model")
    ax1.plot(deciles, [d / 10 for d in deciles], linestyle="--", color="grey", label="Random")
    ax1.set_xlabel("Decile")
    ax1.set_ylabel("% of defaults captured")
    ax1.set_title("Gain Chart")
    ax1.legend(fontsize=8)

    ax2.bar(deciles, [entry["lift"] for entry in lifts], color="#e07b39")
    ax2.axhline(1.0, linestyle="--", color="grey")
    ax2.set_xlabel("Decile")
    ax2.set_ylabel("Lift")
    ax2.set_title("Lift Chart")
    fig.tight_layout()
    return fig


def build_evaluation_report_html(metrics: dict, model_name: str, model_version: str) -> str:
    kpis = kpi_row_html(
        kpi_html("Accuracy", f"{metrics['accuracy']:.1%}"),
        kpi_html("Precision", f"{metrics['precision']:.1%}"),
        kpi_html("Recall", f"{metrics['recall']:.1%}"),
        kpi_html("F1", f"{metrics['f1']:.3f}"),
        kpi_html("ROC-AUC", f"{metrics['roc_auc']:.3f}"),
        kpi_html("PR-AUC", f"{metrics['pr_auc']:.3f}"),
    )

    cm_img = fig_to_base64_img(_confusion_matrix_fig(metrics["confusion_matrix"]), "Confusion matrix")
    calib_img = fig_to_base64_img(_calibration_fig(metrics["calibration_curve"]), "Calibration curve")
    lift_gain_img = fig_to_base64_img(
        _lift_gain_fig(metrics["gain_chart"], metrics["lift_chart"]),
        "Lift and gain charts",
    )
    threshold_table = df_to_html_table(pd.DataFrame(metrics["threshold_analysis"]))

    body = f"""
    <h2>Summary</h2>
    {kpis}
    <p>Evaluated on {metrics['n_samples']:,} labeled reference rows at decision threshold
       <b>{metrics['decision_threshold']:.1%}</b>.</p>

    <h2>Confusion Matrix &amp; Calibration</h2>
    <div style="display:flex; gap:1.5rem; flex-wrap:wrap; align-items:flex-start;">
      <div>{cm_img}</div>
      <div>{calib_img}</div>
    </div>

    <h2>Lift &amp; Gain</h2>
    {lift_gain_img}

    <h2>Threshold Analysis</h2>
    {threshold_table}
    """
    return render_report(
        title="Model Evaluation Report",
        subtitle=f"{model_name} &middot; version {model_version}",
        body_html=body,
    )
