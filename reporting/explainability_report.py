"""
Explainability Report: HTML rendering of a single applicant's SHAP local
explanation, an optional LIME comparison, the plain-English narrative, and
a counterfactual "path to approval" scenario -- all on one page.

Design decision: this is additive alongside (not a replacement for) the
existing PDF applicant report (`explainability/report_generator.py`, served
via `/explain/report`), which stays the printable/mailable adverse-action-
notice format. This HTML version is for screen review and is the only one
of the two that includes the LIME comparison section.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reporting.html_report import df_to_html_table, fig_to_base64_img, kpi_html, kpi_row_html, note_html, render_report


def _shap_waterfall_fig(contributions: list[dict], base_value: float, top_n: int = 10):
    top = contributions[:top_n]
    features = [c["feature"] for c in top][::-1]
    values = [c["shap_value"] for c in top][::-1]
    colors = ["#c0392b" if v > 0 else "#1c7c3f" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(features, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution (impact on predicted default probability)")
    ax.set_title(f"Top factors (base rate: {base_value:.1%})")
    fig.tight_layout()
    return fig


def _lime_bar_fig(contributions: list[dict], top_n: int = 10):
    top = contributions[:top_n]
    if not top:
        return None
    rules = [c["rule"] for c in top][::-1]
    weights = [c["weight"] for c in top][::-1]
    colors = ["#c0392b" if w > 0 else "#1c7c3f" for w in weights]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(rules, weights, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME local linear-surrogate weight")
    ax.set_title("LIME comparison")
    fig.tight_layout()
    return fig


def build_explainability_report_html(
    applicant_id: str,
    predicted_probability: float,
    decision_threshold: float,
    shap_local: dict,
    plain_english_text: str,
    lime_result: dict | None = None,
    counterfactual_changes: list[dict] | None = None,
) -> str:
    decision = "DECLINED" if predicted_probability >= decision_threshold else "APPROVED"
    kpis = kpi_row_html(
        kpi_html("Decision", decision),
        kpi_html("Predicted PD", f"{predicted_probability:.1%}"),
        kpi_html("Threshold", f"{decision_threshold:.1%}"),
    )

    shap_img = fig_to_base64_img(
        _shap_waterfall_fig(shap_local["contributions"], shap_local["base_value"]),
        "SHAP waterfall",
    )

    narrative_html = "".join(f"<p>{line}</p>" for line in plain_english_text.split("\n") if line.strip())

    lime_section = ""
    if lime_result is not None:
        lime_fig = _lime_bar_fig(lime_result.get("contributions", []))
        lime_img = fig_to_base64_img(lime_fig, "LIME comparison") if lime_fig is not None else ""
        agreement_note = note_html(
            "SHAP and LIME explain the same prediction through different mechanisms (exact Shapley "
            "attribution vs. a local linear surrogate) -- differences in which features rank highest are "
            "expected, especially where the model relies on feature interactions a linear surrogate can't "
            "represent, and are worth a reviewer's attention rather than treated as a discrepancy to resolve.",
        )
        lime_section = f"""
        <h2>LIME Comparison</h2>
        {lime_img}
        {agreement_note}
        """

    cf_section = ""
    if counterfactual_changes:
        cf_df = pd.DataFrame(
            [
                {
                    "feature": c["feature"],
                    "from": c["from_value"] if "from_value" in c else c.get("from"),
                    "to": c["to_value"] if "to_value" in c else c.get("to"),
                    "resulting_probability": c["resulting_probability"],
                }
                for c in counterfactual_changes
            ]
        )
        cf_section = f"""
        <h2>What-If Scenario (Path to Approval)</h2>
        {df_to_html_table(cf_df)}
        """
    elif counterfactual_changes is not None:
        cf_section = (
            f"<h2>What-If Scenario</h2>{note_html('No feasible path to approval was found within the search budget.')}"
        )

    body = f"""
    <h2>Summary</h2>
    {kpis}

    <h2>SHAP Local Explanation</h2>
    {shap_img}

    <h2>Plain-English Explanation</h2>
    {narrative_html}

    {lime_section}
    {cf_section}
    """
    return render_report(
        title="Explainability Report",
        subtitle=f"Applicant: {applicant_id}",
        body_html=body,
    )
