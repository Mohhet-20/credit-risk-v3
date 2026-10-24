"""
Data Card (AI Governance): a provenance-and-limitations-focused summary of
the reference dataset -- what it is, where it came from, what it's
appropriate to use for, and its known limitations -- complementary to (not
a duplicate of) `reporting/data_quality_report.py`'s detailed technical
validation report. The Data Quality Report answers "is this data healthy
right now"; the Data Card answers "what IS this data and what should/
shouldn't it be used for", the two governance questions a model card's
"Training & Evaluation Data" section links out to.
"""

from __future__ import annotations

import pandas as pd

from reporting.html_report import df_to_html_table, kpi_html, kpi_row_html, note_html, render_report


def build_data_card_html(result: dict | None) -> str:
    if result is None:
        quality_section = note_html(
            "No data-quality report has been generated yet. POST /data-quality/run first, then this "
            "card's Quality Snapshot section will populate -- provenance and intended-use sections below "
            "are static and don't depend on a report having been run.",
        )
        score_kpis = ""
    else:
        score = result.get("score", {})
        profile = result.get("profile", {})
        score_kpis = kpi_row_html(
            kpi_html("Overall quality score", f"{score.get('overall_score', 'n/a')}"),
            kpi_html("Rows", f"{profile.get('n_rows', 0):,}"),
            kpi_html("Columns", f"{profile.get('n_columns', 0)}"),
            kpi_html("Status", "PASSED" if result.get("passed") else "FAILED"),
        )
        quality_section = (
            f"{score_kpis}"
            f"<p>Full validation detail, missing-value/outlier/duplicate breakdowns, and correlations are in "
            f'the companion <a href="/api/v1/reports/data-quality">Data Quality Report</a>.</p>'
        )

    schema_df = pd.DataFrame(
        [
            {"Category": "Applicant demographics", "Example fields": "age, state, num_dependents, home_ownership"},
            {
                "Category": "Employment & income",
                "Example fields": "employment_status, annual_income, " "employment_length_years",
            },
            {
                "Category": "Credit history",
                "Example fields": "credit_score, num_delinquencies_2yrs, "
                "num_derogatory_marks, revolving_utilization",
            },
            {
                "Category": "Requested loan",
                "Example fields": "loan_amount, loan_term_months, loan_purpose, " "interest_rate",
            },
            {"Category": "Target", "Example fields": "default_flag (binary: repaid vs. defaulted)"},
        ]
    )

    body = f"""
    <h2>Dataset Overview</h2>
    <p>The reference credit-application dataset backing this platform's model training, evaluation, and
    data-quality monitoring (<code>data/raw/credit_applications.csv</code>).</p>

    <h2>Provenance</h2>
    {note_html(
        "This dataset is SYNTHETICALLY GENERATED (see <code>data/generate_synthetic_data.py</code>, "
        "and the Synthetic Data Generator's own documentation), not sourced from real applicants or a "
        "real lending institution. It is statistically shaped to resemble a plausible consumer-credit "
        "population (realistic feature ranges/correlations, an engineered relationship between credit "
        "risk factors and the default label), but it is NOT real-world data and carries none of a real "
        "dataset's genuine demographic, geographic, or macroeconomic structure.",
    )}

    <h2>Feature Summary</h2>
    {df_to_html_table(schema_df)}

    <h2>Quality Snapshot</h2>
    {quality_section}

    <h2>Intended Use</h2>
    <p>Appropriate for: developing and demonstrating this platform's ML pipeline, explainability tooling,
    business-rule engine, and workflow engine end to end. Appropriate for: training/evaluating models used
    ONLY within this platform's own demo/development context.</p>

    <h2>Out-of-Scope Use &amp; Known Limitations</h2>
    {note_html(
        "NOT appropriate as training data for a model making real lending decisions about real people. "
        "Its age, marital-status and state columns are synthetic and causally unrelated to default by "
        "construction, which makes them a clean test bed for the platform's disparate-impact review (see "
        "the Model Card) but says nothing about real applicants. A production deployment requires real, "
        "properly-consented, fair-lending-compliant data, and the review repeated on it.",
    )}
    """
    return render_report(title="Data Card", subtitle="Reference dataset: credit_applications.csv", body_html=body)
