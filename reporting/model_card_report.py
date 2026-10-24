"""
Model Card (AI Governance): a standardized, human-readable summary of a
model version -- what it is, what it was trained/evaluated on, how well it
performs, and its known limitations -- in the spirit of Mitchell et al.
2019's "Model Cards for Model Reporting". This is a REPORT that assembles
data every other module in this platform already computes (model registry
metadata, `reporting/evaluation_report.py`'s metrics, the decision
threshold, and the fair-lending result the training pipeline records in
the registry metadata -- see `analytics/fairness.py` and
`reports/MODEL_VALIDATION.md`). The card reports that result as computed;
it never asserts a fairness outcome the pipeline did not produce.
"""

from __future__ import annotations

import pandas as pd

from reporting.html_report import df_to_html_table, kpi_html, kpi_row_html, note_html, render_report


def _fair_lending_note(metadata: dict) -> str:
    passes = metadata.get("fairness_passes")
    if passes is None:
        return note_html(
            "No fair-lending review is recorded for this model version (it was registered before the "
            "review existed). Re-run the training pipeline to produce one before any adverse-action use."
        )
    outcome = (
        "every age band and marital-status group cleared the four-fifths rule"
        if passes
        else "at least one group fell below the four-fifths rule and is flagged for review"
    )
    return note_html(
        "Fair-lending design: age, marital status and state are excluded from the model's inputs "
        "(ECOA-protected bases and a geographic redlining proxy) and retained only to test decisions for "
        f"disparate impact. On the out-of-time window, {outcome}; per-group approval rates, adverse impact "
        "ratios, observed vs predicted default rates and within-group AUC are in the validation report "
        "(GET /api/v1/models/validation-report). The data is synthetic, so this demonstrates the review "
        "process; a real deployment must repeat it on real applicant data with counsel."
    )


def build_model_card_html(metadata: dict, metrics: dict, *, is_champion: bool) -> str:
    kpis = kpi_row_html(
        kpi_html("ROC-AUC", f"{metrics['roc_auc']:.3f}"),
        kpi_html("PR-AUC", f"{metrics['pr_auc']:.3f}"),
        kpi_html("Precision", f"{metrics['precision']:.1%}"),
        kpi_html("Recall", f"{metrics['recall']:.1%}"),
        kpi_html("Decision Threshold", f"{metrics['decision_threshold']:.1%}"),
        kpi_html("Status", "Champion" if is_champion else "Non-champion"),
    )

    details_df = pd.DataFrame(
        [
            {"Field": "Model name", "Value": metadata.get("model_name", "n/a")},
            {"Field": "Version", "Value": metadata.get("version", "n/a")},
            {"Field": "Registered at", "Value": metadata.get("registered_at", "n/a")},
            {"Field": "Decision threshold", "Value": f"{metadata.get('decision_threshold', 'n/a')}"},
            {"Field": "Evaluated on", "Value": f"{metrics['n_samples']:,} labeled reference rows"},
        ]
    )

    body = f"""
    <h2>Model Details</h2>
    {df_to_html_table(details_df)}

    <h2>Intended Use</h2>
    <p>Estimates the probability that an applicant will default on a consumer credit product, to support
    (not fully automate) a credit approval / rejection / manual-review decision. Intended for use by
    trained loan officers and risk analysts within this platform's workflow (see <code>/workflow/execute</code>),
    subject to the Business Rule Engine's override policies and a human manual-review gate for
    borderline-confidence predictions. NOT intended as the sole basis for an adverse credit action without
    the accompanying explainability report (<code>/reports/explainability</code>) and human review.</p>

    <h2>Training &amp; Evaluation Data</h2>
    <p>Trained and evaluated on this platform's synthetic reference credit-application dataset
    (see <code>data/generate_synthetic_data.py</code> and the accompanying Data Card at
    <code>/reports/data-card</code> for dataset-level detail), split out-of-time: the oldest 60% of
    applications train the model, the next 20% calibrate it and choose the threshold, and the newest 20%
    are held out as the out-of-time test.</p>

    <h2>Evaluation Metrics</h2>
    {kpis}
    <p>Full metric detail (confusion matrix, calibration curve, lift/gain, threshold sweep) is available in
    the companion <a href="/api/v1/reports/evaluation">Model Evaluation Report</a>.</p>

    <h2>Ethical Considerations &amp; Limitations</h2>
    {_fair_lending_note(metadata)}
    {note_html(
        "Predictions are probabilistic estimates, not certainties -- see the calibration curve in the "
        "Model Evaluation Report for how well predicted probabilities track observed outcomes. The Business "
        "Rule Engine and manual-review gate exist specifically so a single miscalibrated prediction cannot "
        "unilaterally decide an application.",
    )}

    <h2>Responsible AI Documentation</h2>
    <p>Every prediction made with this model is: (1) individually explainable via SHAP/LIME
    (<code>/explain</code>, <code>/reports/explainability</code>); (2) subject to admin-editable business
    rules that can override it (<code>/rules</code>); (3) recorded in an immutable audit log
    (<code>/admin/audit-logs</code>); (4) traceable end-to-end through a workflow execution record when run
    via <code>/workflow/execute</code> (<code>/workflow/executions/{{id}}</code>).</p>
    """
    return render_report(
        title="Model Card",
        subtitle=f"{metadata.get('model_name', 'model')} &middot; version {metadata.get('version', 'n/a')}",
        body_html=body,
    )
