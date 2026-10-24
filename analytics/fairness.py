"""
Fair-lending (disparate impact) analysis.

The model never sees ECOA-protected or proxy attributes (`age`,
`marital_status`, `state` -- see `core.constants.MODEL_EXCLUDED_FEATURES`).
Excluding them is necessary but not sufficient: correlated features can
still reproduce a protected-class disparity ("disparate impact"). So the
attributes are kept on the raw record and used *only here*, to test the
model's decisions after the fact -- the standard fair-lending review a bank
runs before a model goes live.

Per group we report:
  - approval rate and the Adverse Impact Ratio (AIR) = group approval rate /
    most-favoured group's approval rate. AIR < 0.80 fails the EEOC
    "four-fifths rule" that regulators use as a screening threshold.
  - observed default rate vs mean predicted PD (calibration within group):
    a model that systematically over-predicts PD for one group penalises it
    even if ranking is fine.
  - ROC-AUC within group (equal discriminatory power).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FOUR_FIFTHS = 0.80

AGE_BINS = [17, 39, 61, 200]
AGE_LABELS = ["18-39", "40-61", "62+"]  # 62+ is Reg B's "elderly" threshold (12 CFR 1002.2(o))


def _group_frame(raw: pd.DataFrame) -> dict[str, pd.Series]:
    groups: dict[str, pd.Series] = {}
    if "age" in raw.columns:
        groups["age_band"] = pd.cut(raw["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    if "marital_status" in raw.columns:
        groups["marital_status"] = raw["marital_status"].astype(str)
    return groups


def disparate_impact_report(raw: pd.DataFrame, y_true, y_proba, threshold: float, min_group_size: int = 30) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    approved = y_proba < threshold
    report: dict = {"threshold": round(float(threshold), 4), "four_fifths_threshold": FOUR_FIFTHS, "attributes": {}}

    for attr, labels in _group_frame(raw.reset_index(drop=True)).items():
        rows = []
        for value in sorted(labels.unique()):
            mask = (labels == value).to_numpy()
            n = int(mask.sum())
            if n < min_group_size:
                continue
            yt, yp = y_true[mask], y_proba[mask]
            rows.append(
                {
                    "group": value,
                    "n": n,
                    "approval_rate": round(float(approved[mask].mean()), 4),
                    "observed_default_rate": round(float(yt.mean()), 4),
                    "mean_predicted_pd": round(float(yp.mean()), 4),
                    "roc_auc": round(float(roc_auc_score(yt, yp)), 4) if 0 < yt.sum() < n else None,
                }
            )
        if not rows:
            continue
        best = max(r["approval_rate"] for r in rows) or 1.0
        for r in rows:
            r["adverse_impact_ratio"] = round(r["approval_rate"] / best, 4)
            r["passes_four_fifths"] = r["adverse_impact_ratio"] >= FOUR_FIFTHS
        report["attributes"][attr] = {
            "groups": rows,
            "min_adverse_impact_ratio": min(r["adverse_impact_ratio"] for r in rows),
            "passes": all(r["passes_four_fifths"] for r in rows),
        }

    report["passes"] = all(a["passes"] for a in report["attributes"].values())
    return report
