"""
Adverse-action reason codes (ECOA / Regulation B, 12 CFR 1002.9).

A US lender that declines an application must tell the applicant the
*principal reasons* for the decision -- "the model said no" is not a
legally sufficient answer, and neither is a raw feature name like
`utilization_x_low_score`. This module turns a local SHAP explanation into
the ranked, plain-language reasons a notice of adverse action carries.

Approach (the "aggregate attributions by reason group" method used in
practice for ML underwriting models):
  1. Every model feature -- raw, engineered, one-hot or target-encoded --
     maps to exactly one *reason group* (e.g. `revolving_utilization`,
     `credit_utilization_band_high` and `utilization_x_low_score` all roll up
     to "Proportion of revolving balances to credit limits is too high").
     Without this, one underlying cause split across three correlated
     features would either crowd out the list or be under-reported.
  2. SHAP values are summed within each group; only groups that pushed risk
     UP (positive contribution) are eligible -- a factor that helped the
     applicant can never be cited as a reason for declining them.
  3. The top N groups (Reg B commentary: "no more than four" is the norm)
     are returned with a stable code, the statement, and the contribution.

Calibration (isotonic/Platt) is a monotone transform of the raw score, so
the ranking of attributions computed on the underlying estimator is the
same ranking that drives the final calibrated PD.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_REASONS = 4


@dataclass(frozen=True)
class ReasonGroup:
    code: str
    statement: str
    features: tuple[str, ...]


# Order matters only for documentation; matching is by longest feature prefix.
REASON_GROUPS: tuple[ReasonGroup, ...] = (
    ReasonGroup("R01", "Credit score is too low", ("credit_score", "credit_score_band")),
    ReasonGroup(
        "R02",
        "Proportion of revolving balances to credit limits is too high",
        ("revolving_utilization", "credit_utilization_band", "utilization_x_low_score", "available_credit"),
    ),
    ReasonGroup("R03", "Debt obligations are too high relative to income", ("debt_to_income_ratio",)),
    ReasonGroup(
        "R04",
        "Requested loan amount is too high relative to income",
        ("loan_to_income_ratio", "payment_to_income_ratio", "monthly_payment_estimate", "loan_amount"),
    ),
    ReasonGroup(
        "R05",
        "Delinquency on past or present credit obligations",
        ("num_delinquencies_2yrs", "months_since_last_delinquency", "has_recent_delinquency", "delinquency_severity"),
    ),
    ReasonGroup("R06", "Derogatory public record or collection action", ("num_derogatory_marks",)),
    ReasonGroup(
        "R07", "Too many recent inquiries for credit", ("num_credit_inquiries_6m", "inquiries_per_open_account")
    ),
    ReasonGroup(
        "R08", "Income is insufficient for the amount of credit requested", ("annual_income", "income_per_dependent")
    ),
    ReasonGroup(
        "R09",
        "Length of employment is insufficient",
        ("employment_length_years", "is_stable_employment", "employment_status"),
    ),
    ReasonGroup(
        "R10",
        "Insufficient liquid assets relative to the loan",
        ("savings_balance", "checking_balance", "total_liquid_assets", "liquidity_to_loan_ratio"),
    ),
    ReasonGroup(
        "R11",
        "Limited or insufficient credit history",
        ("num_open_accounts", "credit_history_depth", "total_credit_limit"),
    ),
    ReasonGroup("R12", "Interest rate and loan terms indicate elevated risk", ("interest_rate", "loan_term_months")),
    ReasonGroup(
        "R13", "Length of residence or housing status", ("years_at_residence", "home_ownership", "is_homeowner")
    ),
    ReasonGroup("R14", "Purpose of the loan", ("loan_purpose",)),
)

OTHER = ReasonGroup("R99", "Other factors in the applicant's credit profile", ())

_PREFIX_INDEX: list[tuple[str, ReasonGroup]] = sorted(
    ((f, g) for g in REASON_GROUPS for f in g.features), key=lambda t: len(t[0]), reverse=True
)


def reason_group_for(feature: str) -> ReasonGroup:
    """Exact match first, then longest prefix (covers one-hot columns such as
    `home_ownership_RENT` and suffixed encodings such as `state_te`)."""
    for prefix, group in _PREFIX_INDEX:
        if feature == prefix or feature.startswith(prefix + "_"):
            return group
    return OTHER


def adverse_action_reasons(contributions: list[dict], max_reasons: int = MAX_REASONS) -> list[dict]:
    """`contributions` is `ModelExplainer.local_explanation()['contributions']`:
    a list of {feature, shap_value, feature_value}. Returns the ranked
    reasons, each with the features that drove it."""
    totals: dict[str, float] = {}
    members: dict[str, list[str]] = {}
    groups: dict[str, ReasonGroup] = {}
    for c in contributions:
        group = reason_group_for(c["feature"])
        groups[group.code] = group
        totals[group.code] = totals.get(group.code, 0.0) + float(c["shap_value"])
        if float(c["shap_value"]) > 0:
            members.setdefault(group.code, []).append(c["feature"])

    ranked = sorted((code for code, v in totals.items() if v > 0), key=lambda code: totals[code], reverse=True)
    return [
        {
            "rank": i + 1,
            "code": code,
            "reason": groups[code].statement,
            "contribution": round(totals[code], 6),
            "features": members.get(code, []),
        }
        for i, code in enumerate(ranked[:max_reasons])
    ]
