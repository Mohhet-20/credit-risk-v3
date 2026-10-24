"""
AI Copilot narrative generation: turns the platform's existing structured
outputs (SHAP/LIME explanations, counterfactuals, evaluation metrics,
monitoring stats, drift reports, data-quality scores) into plain-English
summaries, business insights, and recommended actions.

Design decision (consistent with `explainability/narrative.py`'s existing
precedent): this is deterministic, template-based natural language
generation over already-computed structured data -- NOT a call out to an
external LLM API. Three reasons, same as the original narrative module:
(1) no new runtime dependency, cost, or latency added to a request; (2) no
new secret/API key to provision, which would make this feature silently
non-functional in exactly the kind of offline/sandboxed/air-gapped
deployment this platform is designed to support out of the box; (3) for a
regulated credit-risk product, a reproducible, auditable explanation is a
requirement, not a nice-to-have -- the same input always produces the same
"copilot" narrative, which matters if a business user ever needs to explain
*how* the copilot reached its summary. A real LLM-backed copilot (for freer-
form Q&A) is a legitimate future enhancement and is called out explicitly
as such in docs/SECURITY.md and docs/ARCHITECTURE.md rather than faked here.

Every function in this module is a pure function: structured dict in,
structured dict out (`{"headline": str, "narrative": str, "insights":
list[str], "recommended_actions": list[str]}`), with no I/O and no
dependency on FastAPI, a database session, or a model bundle -- callers
(the `/copilot` router, tests) supply already-fetched data. This keeps the
module trivially unit-testable and reusable outside the API layer (e.g. a
future scheduled job that emails a daily executive briefing).
"""

from __future__ import annotations


def _copilot_response(headline: str, narrative: str, insights: list[str], actions: list[str]) -> dict:
    return {"headline": headline, "narrative": narrative, "insights": insights, "recommended_actions": actions}


def explain_prediction_for_copilot(
    plain_english_text: str,
    predicted_probability: float,
    decision_threshold: float,
    shap_contributions: list[dict],
    counterfactual_changes: list[dict] | None,
    top_n: int = 3,
) -> dict:
    """Wraps the existing `explain_prediction_plain_english()` narrative
    (SHAP-based, already deterministic) with a copilot-style headline and
    concrete recommended actions, optionally informed by a counterfactual
    "path to approval" if one was computed."""
    decision = "declined" if predicted_probability >= decision_threshold else "approved"
    margin = abs(predicted_probability - decision_threshold)
    confidence = "clear-cut" if margin > 0.15 else "borderline"

    headline = (
        f"This application was {decision} ({confidence}) at a {predicted_probability:.1%} estimated default "
        f"probability against a {decision_threshold:.1%} threshold."
    )

    top_risk_factors = [c for c in shap_contributions if c["shap_value"] > 0][:top_n]
    insights = [
        (
            f"'{c['feature']}' (value: {c['feature_value']}) was the largest risk-increasing factor."
            if i == 0
            else f"'{c['feature']}' (value: {c['feature_value']}) also increased predicted risk."
        )
        for i, c in enumerate(top_risk_factors)
    ]
    if not insights:
        insights = ["No individual factor pushed risk up significantly -- this profile is broadly low-risk."]

    actions: list[str] = []
    if decision == "declined":
        if counterfactual_changes:
            change = counterfactual_changes[0]
            actions.append(
                f"A feasible path to approval exists: adjusting '{change['feature']}' from "
                f"{change.get('from_value', change.get('from'))} to {change.get('to_value', change.get('to'))} "
                f"would materially reduce predicted risk (see the full counterfactual scenario)."
            )
        else:
            actions.append(
                "No feasible single-feature path to approval was found within the search budget -- "
                "consider a manual underwriting review if this applicant has compensating factors "
                "not captured by the model's input features."
            )
        actions.append("Route to manual review if the applicant disputes the automated decision.")
    else:
        actions.append("No action required -- proceed with standard approval workflow.")
        if confidence == "borderline":
            actions.append(
                "This approval is borderline (close to the decision threshold) -- a manual second look "
                "is reasonable for high loan-amount applications."
            )

    return _copilot_response(headline, plain_english_text, insights, actions)


def summarize_model_performance(metrics: dict, model_name: str, model_version: str) -> dict:
    headline = (
        f"{model_name} (version {model_version}) is scoring {metrics['accuracy']:.1%} accuracy and "
        f"{metrics['roc_auc']:.3f} ROC-AUC on the latest evaluation."
    )
    narrative = (
        f"Evaluated on {metrics['n_samples']:,} labeled reference rows at a {metrics['decision_threshold']:.1%} "
        f"decision threshold: precision {metrics['precision']:.1%}, recall {metrics['recall']:.1%}, "
        f"F1 {metrics['f1']:.3f}, PR-AUC {metrics['pr_auc']:.3f}."
    )

    insights = []
    if metrics["precision"] < 0.6:
        insights.append(
            "Precision is relatively low -- a meaningful share of declined applications may be false "
            "positives, worth a manual sample review."
        )
    if metrics["recall"] < 0.6:
        insights.append("Recall is relatively low -- the model may be missing a meaningful share of true defaults.")
    if metrics["roc_auc"] >= 0.8:
        insights.append("Discrimination (ROC-AUC) is strong -- the model separates risk tiers well overall.")
    elif metrics["roc_auc"] < 0.65:
        insights.append("Discrimination (ROC-AUC) is weak -- consider revisiting features or trying another model.")
    if not insights:
        insights.append("Metrics are within a normal, healthy range -- no immediate performance concerns.")

    actions = []
    if metrics["roc_auc"] < 0.65 or metrics["precision"] < 0.5:
        actions.append("Consider retraining with an expanded feature set or a different model family.")
    else:
        actions.append("No retraining action needed based on current metrics alone.")
    actions.append("Cross-check against the drift report before concluding performance is stable long-term.")

    return _copilot_response(headline, narrative, insights, actions)


def summarize_monitoring(stats: dict, distribution: dict, drift: dict, latency: dict) -> dict:
    n = stats.get("n_predictions", 0)
    avg_proba = stats.get("avg_predicted_probability")
    window_days = stats.get("window_days", 30)

    headline = f"{n:,} predictions served in the last {window_days} days" + (
        f", averaging a {avg_proba:.1%} predicted default probability." if avg_proba is not None else "."
    )

    narrative_parts = [headline]
    if latency.get("n", 0) > 0:
        narrative_parts.append(
            f"Latency: p50 {latency['p50_ms']:.0f}ms, p95 {latency['p95_ms']:.0f}ms, p99 {latency['p99_ms']:.0f}ms."
        )
    decision_counts = distribution.get("decision_counts", {})
    if decision_counts:
        total = sum(decision_counts.values()) or 1
        approval_rate = decision_counts.get("approved", 0) / total
        narrative_parts.append(f"Approval rate over this window: {approval_rate:.1%}.")

    insights = []
    severity = drift.get("overall_severity")
    if severity in ("high", "severe", "critical"):
        insights.append(f"Feature drift severity is '{severity}' -- input data distribution has shifted meaningfully.")
    elif severity in ("moderate", "medium"):
        insights.append(f"Feature drift severity is '{severity}' -- worth monitoring, not yet critical.")
    elif severity in ("low", "none", "stable"):
        insights.append("Feature drift is low/stable -- input data still resembles the training distribution.")
    if latency.get("n", 0) > 0 and latency.get("p95_ms", 0) > 1000:
        insights.append("p95 latency exceeds 1 second -- worth investigating if this is a new regression.")
    if n == 0:
        insights.append("No predictions recorded in this window -- confirm the scoring endpoint is receiving traffic.")
    if not insights:
        insights.append("No monitoring anomalies detected in this window.")

    actions = []
    if severity in ("high", "severe", "critical"):
        actions.append("Investigate the highest-PSI features in the drift report and consider retraining.")
    if latency.get("n", 0) > 0 and latency.get("p95_ms", 0) > 1000:
        actions.append("Review recent deploys or infrastructure changes that could explain the latency increase.")
    if not actions:
        actions.append("No action required -- continue routine monitoring.")

    return _copilot_response(headline, " ".join(narrative_parts), insights, actions)


def summarize_data_quality(report: dict) -> dict:
    score = report.get("score", {})
    overall = score.get("overall_score")
    passed = report.get("passed")

    headline = (
        f"Latest data-quality run: overall score {overall:.1f}/100, "
        f"{'PASSED' if passed else 'FAILED'} the hard-gate checks."
        if overall is not None
        else "Data-quality score unavailable for the latest run."
    )

    narrative = (
        f"Validity {score.get('validity_score', 'n/a')}, completeness {score.get('completeness_score', 'n/a')}, "
        f"uniqueness {score.get('uniqueness_score', 'n/a')}, "
        f"{score.get('n_validation_issues', 0)} validation issue(s) found."
    )

    insights = []
    if not passed:
        insights.append(
            "This dataset failed one or more critical validation gates -- treat downstream results with caution."
        )
    if score.get("completeness_score", 100) < 80:
        insights.append("Completeness score is low -- a meaningful share of values are missing.")
    if score.get("freshness_hours") is not None and score["freshness_hours"] > 24 * 7:
        insights.append("The underlying dataset file hasn't been updated in over a week.")
    if not insights:
        insights.append("No data-quality concerns detected in the latest run.")

    actions = []
    if not passed:
        actions.append("Do not promote a model trained on this data until the failing checks are resolved.")
        actions.append("Review the validation issue list and correct or exclude the offending records.")
    else:
        actions.append("No action required -- data quality is within acceptable bounds.")

    return _copilot_response(headline, narrative, insights, actions)


def summarize_governance(dashboard: dict) -> dict:
    """Turns `GET /reports/compliance-dashboard`'s aggregated numbers
    (model governance status, business-rule inventory, recent workflow
    outcomes, data-quality status) into an executive-readable governance
    summary -- the "Executive Insights" extension for the AI
    Governance work. Same deterministic-template approach as every other
    `summarize_*` function in this module, over data the Compliance
    Dashboard already computes (no new metrics invented here)."""
    mg = dashboard.get("model_governance", {})
    rules = dashboard.get("business_rules", {})
    workflow = dashboard.get("workflow", {})
    dq = dashboard.get("data_quality", {})

    champion = mg.get("champion_version")
    headline = (
        f"Champion model {champion or '(none registered)'} governing {workflow.get('total', 0)} workflow "
        f"execution(s) in the last {workflow.get('window_days', 30)} days, enforced by "
        f"{rules.get('active_rules', 0)} active business rule(s)."
    )

    narrative_parts = [headline]
    if workflow.get("total", 0) > 0:
        failure_rate = workflow.get("failed", 0) / workflow["total"]
        narrative_parts.append(f"Workflow failure rate: {failure_rate:.1%}.")
    if dq.get("has_report"):
        narrative_parts.append(
            f"Latest data-quality run: {'PASSED' if dq.get('latest_report_passed') else 'FAILED'} "
            f"(score {dq.get('latest_overall_score', 'n/a')})."
        )

    insights = []
    if champion is None:
        insights.append("No champion model is currently registered -- prediction/workflow endpoints will fail.")
    if rules.get("active_rules", 0) == 0:
        insights.append(
            "No active business rules -- every decision is currently made by the model alone, with no "
            "policy override layer."
        )
    if workflow.get("total", 0) > 0 and workflow.get("failed", 0) / workflow["total"] > 0.1:
        insights.append(
            "Workflow failure rate exceeds 10% -- investigate recent execution failures before this becomes routine."
        )
    if dq.get("has_report") and not dq.get("latest_report_passed"):
        insights.append(
            "The latest data-quality run FAILED its hard-gate checks -- treat recent predictions with caution."
        )
    if not insights:
        insights.append(
            "No governance concerns detected -- champion model registered, rules active, workflows healthy."
        )

    actions = []
    if champion is None:
        actions.append("Promote a model version to champion before accepting live traffic.")
    if rules.get("active_rules", 0) == 0:
        actions.append("Consider defining at least a baseline manual-review band rule for borderline predictions.")
    if not dq.get("has_report"):
        actions.append("Run a data-quality check (POST /data-quality/run) to establish a governance baseline.")
    if not actions:
        actions.append("No governance action required -- continue routine oversight.")

    return _copilot_response(headline, " ".join(narrative_parts), insights, actions)


def generate_executive_summary(
    model_summary: dict | None,
    monitoring_summary: dict | None,
    data_quality_summary: dict | None,
    governance_summary: dict | None = None,
) -> dict:
    """Combines the domain summaries above into one executive-level
    briefing -- the "ask the copilot for today's status" endpoint. Any
    section can be `None` if that data isn't available yet (e.g. no
    predictions served, no data-quality run yet); the summary degrades
    gracefully rather than failing. `governance_summary` (from
    `summarize_governance()`) is optional and keyword-defaulted so existing
    callers that predate the AI Governance layer keep working
    unchanged."""
    sections = []
    all_insights: list[str] = []
    all_actions: list[str] = []

    if model_summary is not None:
        sections.append(f"MODEL PERFORMANCE: {model_summary['headline']}")
        all_insights.extend(model_summary["insights"])
        all_actions.extend(model_summary["recommended_actions"])
    if monitoring_summary is not None:
        sections.append(f"OPERATIONS: {monitoring_summary['headline']}")
        all_insights.extend(monitoring_summary["insights"])
        all_actions.extend(monitoring_summary["recommended_actions"])
    if data_quality_summary is not None:
        sections.append(f"DATA QUALITY: {data_quality_summary['headline']}")
        all_insights.extend(data_quality_summary["insights"])
        all_actions.extend(data_quality_summary["recommended_actions"])
    if governance_summary is not None:
        sections.append(f"GOVERNANCE: {governance_summary['headline']}")
        all_insights.extend(governance_summary["insights"])
        all_actions.extend(governance_summary["recommended_actions"])

    if not sections:
        return _copilot_response(
            "No data available yet for an executive summary.",
            "Run a prediction, a data-quality check, and ensure a champion model is registered, then retry.",
            [],
            [],
        )

    headline = f"Executive briefing across {len(sections)} area(s): " + "; ".join(s.split(": ", 1)[1] for s in sections)
    narrative = "\n".join(sections)
    # De-duplicate while preserving order -- "no action required" showing up
    # three times across sections is noise, not signal.
    actions = list(dict.fromkeys(a for a in all_actions if "no action" not in a.lower()))
    if not actions:
        actions = ["No action required across any monitored area -- the platform is healthy."]

    return _copilot_response(headline, narrative, list(dict.fromkeys(all_insights)), actions)
