"""
Synthetic credit-application data generator.

Why synthetic data: real consumer credit data is regulated (FCRA/GDPR) and
essentially never shareable in a public portfolio project. Instead of a toy
`make_classification` blob, this generator builds a *causally structured*
dataset — the default probability is an explicit logistic function of
interpretable underwriting features (credit score, DTI, utilization,
delinquencies, income, loan-to-income, employment stability) plus noise.
This matters for the rest of the platform: SHAP explanations, PDP/ICE plots,
and calibration curves are only meaningful to review if the ground-truth
relationship between features and risk is realistic and recoverable — a
data-leakage-free target lets us honestly demonstrate ROC-AUC ~0.80-0.90
rather than a suspicious 0.99 that would come from a trivially separable
synthetic label.

Injected realism:
- MCAR/MAR missingness on a subset of columns (income, employment length,
  utilization) to exercise the missing-value-strategy module.
- Heavy-tailed outliers on income/loan_amount to exercise outlier handling.
- Mild label noise (a fraction of borderline cases flip) so the ceiling
  ROC-AUC is realistic (~0.85-0.92) rather than a trivial 1.0.
- A near-duplicate injection so the data-quality module has something to find.
- Mild macro drift: the baseline default propensity rises slowly across the
  two-year application window (a deteriorating credit cycle), so the
  out-of-time test window in `pipelines/run_training.py` is a genuine test of
  stability rather than a reshuffle of the training distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_credit_dataset(n_samples: int = 50_000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_samples

    # --- Demographics / application identifiers ---
    customer_id = np.array([f"CUST-{i:07d}" for i in range(n)])
    application_id = np.array([f"APP-{i:07d}" for i in range(n)])
    day_offset = rng.integers(0, 730, size=n)
    application_date = pd.to_datetime("2023-01-01") + pd.to_timedelta(day_offset, unit="D")

    age = np.clip(rng.normal(40, 12, n), 18, 80).round().astype(int)
    num_dependents = rng.poisson(1.1, n).clip(0, 6)
    years_at_residence = np.clip(rng.exponential(4, n), 0, 30).round(1)

    home_ownership = rng.choice(["OWN", "MORTGAGE", "RENT", "OTHER"], size=n, p=[0.18, 0.42, 0.36, 0.04])
    marital_status = rng.choice(["single", "married", "divorced", "widowed"], size=n, p=[0.38, 0.45, 0.13, 0.04])
    education_level = rng.choice(
        ["high_school", "bachelors", "masters", "doctorate", "other"],
        size=n,
        p=[0.32, 0.38, 0.18, 0.05, 0.07],
    )
    employment_status = rng.choice(
        ["employed", "self_employed", "unemployed", "retired"],
        size=n,
        p=[0.68, 0.14, 0.06, 0.12],
    )
    state = rng.choice(["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"], size=n)
    loan_purpose = rng.choice(
        [
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "major_purchase",
            "medical",
            "small_business",
            "car",
            "other",
        ],
        size=n,
        p=[0.28, 0.22, 0.12, 0.10, 0.08, 0.08, 0.07, 0.05],
    )

    # --- Financial features (correlated with age/employment for realism) ---
    base_income = rng.lognormal(mean=10.8, sigma=0.55, size=n)  # ~ centered ~50k
    employment_length_years = np.clip(
        rng.exponential(5, n) + np.where(employment_status == "retired", 15, 0), 0, 45
    ).round(1)
    annual_income = np.clip(base_income * (1 + 0.01 * np.clip(age - 25, 0, None)), 12_000, 500_000).round(2)
    annual_income[employment_status == "unemployed"] *= rng.uniform(0.2, 0.5, (employment_status == "unemployed").sum())

    credit_score = (
        np.clip(
            rng.normal(680, 75, n)
            + 0.15 * (employment_length_years - 5)
            - 0.0004 * np.clip(200_000 - annual_income, 0, None),
            300,
            850,
        )
        .round()
        .astype(int)
    )

    revolving_utilization = np.clip(rng.beta(2, 3, n) * (1 - credit_score / 1200), 0, 1).round(3)
    total_credit_limit = np.clip(rng.lognormal(9.2, 0.7, n), 500, 200_000).round(2)
    num_open_accounts = np.clip(rng.poisson(6, n) + (annual_income > 80_000).astype(int) * 2, 0, 30)
    num_credit_inquiries_6m = rng.poisson(1.2, n).clip(0, 15)
    num_delinquencies_2yrs = rng.poisson(np.clip(0.6 - credit_score / 2000, 0.02, None), n).clip(0, 10)
    num_derogatory_marks = rng.poisson(np.clip(0.3 - credit_score / 3000, 0.01, None), n).clip(0, 8)
    months_since_last_delinquency = np.where(num_delinquencies_2yrs > 0, rng.integers(1, 60, n), 999)

    debt_to_income_ratio = np.clip(rng.beta(2, 5, n) * 60 + 0.05 * num_open_accounts, 0, 65).round(2)
    savings_balance = np.clip(rng.lognormal(7.5, 1.4, n) * (annual_income / 50_000), 0, 300_000).round(2)
    checking_balance = np.clip(rng.lognormal(6.5, 1.2, n), 0, 100_000).round(2)

    loan_amount = np.clip(rng.lognormal(9.0, 0.6, n), 1000, 100_000).round(2)
    loan_term_months = rng.choice([12, 24, 36, 48, 60, 72], size=n, p=[0.05, 0.15, 0.35, 0.2, 0.15, 0.1])
    interest_rate = np.clip(5.0 + (850 - credit_score) / 25 + rng.normal(0, 1.2, n), 3.5, 35.0).round(2)

    # --- Ground-truth default probability (causal structure) ---
    z = (
        -6.2
        + 0.018 * (700 - credit_score)
        + 0.045 * debt_to_income_ratio
        + 2.6 * revolving_utilization
        + 0.55 * num_delinquencies_2yrs
        + 0.9 * num_derogatory_marks
        + 0.10 * num_credit_inquiries_6m
        + 0.9 * (loan_amount / np.maximum(annual_income, 1000))
        + 0.05 * interest_rate
        - 0.012 * employment_length_years
        - 0.000006 * annual_income
        + 0.7 * (employment_status == "unemployed").astype(float)
        + 0.35 * (home_ownership == "RENT").astype(float)
        - 0.25 * (home_ownership == "OWN").astype(float)
        + 0.25 * (day_offset / 730)  # macro drift: later vintages default slightly more
    )
    pd_true = _sigmoid(z)
    default_flag = (rng.uniform(0, 1, n) < pd_true).astype(int)

    # Label noise: flip ~2% of borderline (0.3-0.7 predicted-prob) cases.
    borderline = (pd_true > 0.3) & (pd_true < 0.7)
    flip_idx = rng.choice(np.where(borderline)[0], size=int(0.02 * borderline.sum()), replace=False)
    default_flag[flip_idx] = 1 - default_flag[flip_idx]

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "application_id": application_id,
            "application_date": application_date,
            "age": age,
            "annual_income": annual_income,
            "employment_length_years": employment_length_years,
            "employment_status": employment_status,
            "home_ownership": home_ownership,
            "marital_status": marital_status,
            "education_level": education_level,
            "state": state,
            "num_dependents": num_dependents,
            "years_at_residence": years_at_residence,
            "loan_amount": loan_amount,
            "loan_purpose": loan_purpose,
            "loan_term_months": loan_term_months,
            "interest_rate": interest_rate,
            "credit_score": credit_score,
            "debt_to_income_ratio": debt_to_income_ratio,
            "num_open_accounts": num_open_accounts,
            "num_credit_inquiries_6m": num_credit_inquiries_6m,
            "revolving_utilization": revolving_utilization,
            "num_delinquencies_2yrs": num_delinquencies_2yrs,
            "months_since_last_delinquency": months_since_last_delinquency,
            "num_derogatory_marks": num_derogatory_marks,
            "total_credit_limit": total_credit_limit,
            "savings_balance": savings_balance,
            "checking_balance": checking_balance,
            "default_flag": default_flag,
        }
    )

    # --- Inject realistic missingness (MAR: more likely missing for self-employed/unemployed) ---
    for col, base_rate in [
        ("annual_income", 0.03),
        ("employment_length_years", 0.04),
        ("revolving_utilization", 0.02),
        ("years_at_residence", 0.05),
        ("debt_to_income_ratio", 0.02),
    ]:
        miss_prob = np.where(df["employment_status"].isin(["self_employed", "unemployed"]), base_rate * 4, base_rate)
        mask = rng.uniform(0, 1, n) < miss_prob
        df.loc[mask, col] = np.nan

    # --- Inject a few extreme outliers to exercise outlier handling ---
    outlier_idx = rng.choice(n, size=max(5, n // 500), replace=False)
    df.loc[outlier_idx, "annual_income"] = df.loc[outlier_idx, "annual_income"] * rng.uniform(5, 15, len(outlier_idx))
    outlier_idx2 = rng.choice(n, size=max(5, n // 500), replace=False)
    df.loc[outlier_idx2, "loan_amount"] = df.loc[outlier_idx2, "loan_amount"] * rng.uniform(3, 6, len(outlier_idx2))

    # --- Inject exact duplicate records for the data-quality module to catch ---
    n_dupes = max(3, n // 1000)
    dupe_rows = df.sample(n=n_dupes, random_state=seed).copy()
    dupe_rows["customer_id"] = dupe_rows["customer_id"] + "-DUP"
    df = pd.concat([df, dupe_rows], ignore_index=True)

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic credit risk dataset.")
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "raw" / "credit_applications.csv"),
    )
    args = parser.parse_args()

    df = generate_credit_dataset(n_samples=args.n_samples, seed=args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(f"Default rate: {df['default_flag'].mean():.3%}")


if __name__ == "__main__":
    main()
