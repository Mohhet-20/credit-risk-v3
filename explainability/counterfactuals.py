"""
Counterfactual scenario generator ("what would this applicant need to change
to be approved?").

Design decision: rather than pulling in a heavyweight library (e.g. DiCE,
which needs a differentiable model or a slow genetic search), we implement a
constrained greedy coordinate-descent search over a small, explicitly
"actionable" feature whitelist — the fields an applicant could realistically
change (pay down revolving balances, increase income, reduce requested loan
amount) as opposed to immutable ones (age) or fields that aren't really
actionable on a short horizon (credit_score itself, which is a downstream
consequence of the actionable fields, not a lever). At each step we try a
fixed perturbation on every actionable feature, keep whichever single change
most reduces predicted risk, and repeat until the decision threshold is
cleared or the step budget is exhausted. This is intentionally simple and
auditable — a credit committee can read the resulting change-list directly —
rather than a black-box optimizer output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.logging import get_logger

logger = get_logger(__name__)

# feature -> (step_size, direction, min_bound, max_bound)
# direction: -1 means "decreasing this feature reduces risk", +1 means "increasing reduces risk"
ACTIONABLE_FEATURES: dict[str, dict] = {
    "revolving_utilization": {"step": 0.05, "direction": -1, "min": 0.0, "max": 1.0},
    "loan_amount": {"step": 500.0, "direction": -1, "min": 1000.0, "max": None},
    "debt_to_income_ratio": {"step": 2.0, "direction": -1, "min": 0.0, "max": None},
    "savings_balance": {"step": 500.0, "direction": 1, "min": 0.0, "max": None},
    "checking_balance": {"step": 250.0, "direction": 1, "min": 0.0, "max": None},
    "num_credit_inquiries_6m": {"step": 1.0, "direction": -1, "min": 0.0, "max": None},
}


@dataclass
class CounterfactualResult:
    achieved: bool
    original_probability: float
    final_probability: float
    changes: list[dict] = field(default_factory=list)
    n_steps: int = 0


def generate_counterfactual(
    predict_fn,
    feature_pipeline,
    applicant_raw: pd.DataFrame,
    threshold: float,
    max_steps: int = 30,
) -> CounterfactualResult:
    """`predict_fn`: callable(transformed_df) -> array of P(default).
    `feature_pipeline`: fitted FeaturePipeline (raw -> model-ready features).
    `applicant_raw`: single-row raw (pre-transform) dataframe for one applicant.
    """
    current = applicant_raw.copy().reset_index(drop=True)
    original_proba = float(predict_fn(feature_pipeline.transform(current))[0])

    if original_proba < threshold:
        return CounterfactualResult(
            achieved=True, original_probability=original_proba, final_probability=original_proba, changes=[], n_steps=0
        )

    changes_log: list[dict] = []
    proba = original_proba

    for _step in range(max_steps):
        if proba < threshold:
            break

        best_feature, best_new_value, best_proba = None, None, proba
        for feat, spec in ACTIONABLE_FEATURES.items():
            if feat not in current.columns:
                continue
            trial = current.copy()
            new_val = trial.loc[0, feat] + spec["direction"] * spec["step"]
            if spec["min"] is not None:
                new_val = max(new_val, spec["min"])
            if spec["max"] is not None:
                new_val = min(new_val, spec["max"])
            if new_val == trial.loc[0, feat]:
                continue
            trial.loc[0, feat] = new_val
            trial_proba = float(predict_fn(feature_pipeline.transform(trial))[0])
            if trial_proba < best_proba:
                best_feature, best_new_value, best_proba = feat, new_val, trial_proba

        if best_feature is None:
            logger.info("Counterfactual search stalled: no single-feature change improves risk further.")
            break

        old_value = current.loc[0, best_feature]
        current.loc[0, best_feature] = best_new_value
        changes_log.append(
            {"feature": best_feature, "from": old_value, "to": best_new_value, "resulting_probability": best_proba}
        )
        proba = best_proba

    return CounterfactualResult(
        achieved=proba < threshold,
        original_probability=original_proba,
        final_probability=proba,
        changes=changes_log,
        n_steps=len(changes_log),
    )
