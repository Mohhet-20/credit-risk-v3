"""
Loan structure recommendation engine.

Design decision: this is deliberately scoped to the two levers a loan
officer can realistically renegotiate with an applicant -- `loan_amount` and
`loan_term_months` -- rather than the fully unconstrained counterfactual
search in `explainability.counterfactuals` (which can suggest changing
immutable/unrealistic fields like credit_score or age). A small, explainable
grid search over realistic amount/term combinations produces recommendations
a loan officer can actually act on in a conversation with the applicant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

AMOUNT_MULTIPLIERS = [1.0, 0.85, 0.7, 0.5]
TERM_OPTIONS_MONTHS = [24, 36, 48, 60, 72]
MIN_LOAN_AMOUNT = 500.0


@dataclass
class LoanOption:
    loan_amount: float
    loan_term_months: int
    predicted_probability: float
    estimated_monthly_payment: float
    approved: bool


def _monthly_payment(amount: float, annual_rate_pct: float, term_months: int) -> float:
    monthly_rate = (annual_rate_pct / 100) / 12
    if monthly_rate <= 0:
        return amount / term_months
    return amount * monthly_rate / (1 - (1 + monthly_rate) ** (-term_months))


def recommend_loan_structures(
    predict_fn: Callable[[pd.DataFrame], pd.Series],
    feature_pipeline,
    application_raw: pd.DataFrame,
    decision_threshold: float,
    top_n: int = 3,
) -> list[LoanOption]:
    """Grid-search loan_amount x loan_term_months and return the top-N
    approved structures, ranked to favor the highest amount (closest to what
    the applicant originally wanted) and, as a tiebreaker, the lowest
    monthly payment."""
    base_row = application_raw.iloc[0]
    original_amount = float(base_row["loan_amount"])
    interest_rate = float(base_row["interest_rate"])

    candidates: list[LoanOption] = []
    for mult in AMOUNT_MULTIPLIERS:
        candidate_amount = round(original_amount * mult, -2)  # nearest $100
        if candidate_amount < MIN_LOAN_AMOUNT:
            continue
        for term in TERM_OPTIONS_MONTHS:
            row = application_raw.copy()
            row["loan_amount"] = candidate_amount
            row["loan_term_months"] = term

            transformed = feature_pipeline.transform(row)
            proba = float(predict_fn(transformed)[0])
            approved = proba < decision_threshold

            candidates.append(
                LoanOption(
                    loan_amount=candidate_amount,
                    loan_term_months=term,
                    predicted_probability=proba,
                    estimated_monthly_payment=round(_monthly_payment(candidate_amount, interest_rate, term), 2),
                    approved=approved,
                )
            )

    approved_candidates = [c for c in candidates if c.approved]
    ranked = sorted(approved_candidates, key=lambda c: (-c.loan_amount, c.estimated_monthly_payment))
    return ranked[:top_n]
