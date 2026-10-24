"""
Data Quality Report: HTML rendering of a `DataQualityService.run_and_persist()`
result (validation issues + profiling statistics + the composite quality
score), reusing the same `reporting/html_report.py` scaffolding as the
model/evaluation/drift reports from the previous pass.
"""

from __future__ import annotations

import pandas as pd

from reporting.html_report import df_to_html_table, kpi_html, kpi_row_html, note_html, render_report

_SEVERITY_BADGE = {"critical": "🔴", "warning": "🟠", "info": "🔵"}


def build_data_quality_report_html(result: dict) -> str:
    score = result.get("score", {})
    validation = result.get("validation", {})
    profile = result.get("profile", {})

    kpis = kpi_row_html(
        kpi_html("Overall score", f"{score.get('overall_score', 'n/a')}"),
        kpi_html("Validity", f"{score.get('validity_score', 'n/a')}"),
        kpi_html("Completeness", f"{score.get('completeness_score', 'n/a')}"),
        kpi_html("Uniqueness", f"{score.get('uniqueness_score', 'n/a')}"),
        kpi_html("Status", "PASSED" if result.get("passed") else "FAILED"),
    )

    freshness = score.get("freshness_hours")
    freshness_note = note_html(f"Reference dataset is {freshness:.1f} hours old.") if freshness is not None else ""

    issues = validation.get("issues", [])
    if issues:
        issues_df = pd.DataFrame(
            [
                {
                    "severity": _SEVERITY_BADGE.get(i["severity"], "") + " " + i["severity"],
                    "category": i["category"],
                    "message": i["message"],
                }
                for i in issues
            ]
        )
        issues_html = df_to_html_table(issues_df)
    else:
        issues_html = note_html("No validation issues found.")

    missing_summary = profile.get("missing_value_summary", {})
    missing_html = (
        df_to_html_table(pd.DataFrame(missing_summary.get("worst_columns", [])))
        if missing_summary.get("worst_columns")
        else note_html("No missing values detected.")
    )

    dup = profile.get("duplicate_analysis", {})
    dup_html = kpi_row_html(
        kpi_html("Duplicate rows", str(dup.get("n_exact_duplicate_rows", 0))),
        kpi_html("% of dataset", f"{dup.get('pct_duplicate_rows', 0):.1%}"),
        kpi_html("Duplicate groups", str(dup.get("n_distinct_duplicate_groups", 0))),
    )

    outliers = profile.get("outlier_summary", [])
    outliers_html = (
        df_to_html_table(pd.DataFrame(outliers).sort_values("pct_outliers", ascending=False).head(10))
        if outliers
        else note_html("No outlier summary available.")
    )

    cardinality = profile.get("cardinality", [])
    high_card = [c for c in cardinality if c.get("is_high_cardinality")]
    cardinality_html = (
        df_to_html_table(pd.DataFrame(high_card))
        if high_card
        else note_html("No high-cardinality categorical columns detected.")
    )

    correlations = profile.get("correlations", [])
    correlations_html = (
        df_to_html_table(pd.DataFrame(correlations).head(10))
        if correlations
        else note_html("Not enough numeric columns to compute correlations.")
    )

    business_summary = profile.get("business_summary", {})
    business_html = df_to_html_table(pd.DataFrame([business_summary])) if business_summary else ""

    body = f"""
    <h2>Summary</h2>
    {kpis}
    {freshness_note}
    <p>{profile.get('n_rows', 0):,} rows &times; {profile.get('n_columns', 0)} columns
       ({validation.get('n_issues', 0)} validation issue(s) found).</p>

    <h2>Validation Issues</h2>
    {issues_html}

    <h2>Missing Values</h2>
    {missing_html}

    <h2>Duplicates</h2>
    {dup_html}

    <h2>Outliers (IQR method, top 10)</h2>
    {outliers_html}

    <h2>High-Cardinality Columns</h2>
    {cardinality_html}

    <h2>Top Feature Correlations</h2>
    {correlations_html}

    <h2>Business Summary</h2>
    {business_html}
    """
    return render_report(
        title="Data Quality Report",
        subtitle=f"Dataset: {result.get('dataset_label', 'reference')}",
        body_html=body,
    )
