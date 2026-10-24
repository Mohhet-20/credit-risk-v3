"""
Plain-English prediction explainer.

Design decision: rather than calling out to an LLM API (adding a runtime
dependency, cost, and latency to every prediction explanation, plus a
non-deterministic explanation for a regulated decision), this is a
deterministic template-based natural language generator built directly from
the SHAP local-explanation contributions and the feature catalog's human
descriptions. For a credit decision, an auditable, reproducible explanation
is a feature, not a limitation — the same input always produces the same
wording, which matters for adverse-action-notice style compliance use cases.
"""

from __future__ import annotations

from feature_store.catalog import catalog


def explain_prediction_plain_english(
    local_explanation: dict,
    predicted_probability: float,
    decision_threshold: float,
    top_n: int = 5,
) -> str:
    decision = "declined" if predicted_probability >= decision_threshold else "approved"
    contributions = local_explanation["contributions"][:top_n]

    lines = [
        f"This application was {decision} with an estimated default probability of "
        f"{predicted_probability:.1%} against a decision threshold of {decision_threshold:.1%}.",
        "",
        "The main factors behind this prediction were:",
    ]

    for i, c in enumerate(contributions, start=1):
        direction = "increased" if c["shap_value"] > 0 else "decreased"
        magnitude = abs(c["shap_value"])
        strength = "strongly" if magnitude > 0.05 else ("moderately" if magnitude > 0.02 else "slightly")
        description = catalog.describe(c["feature"])
        lines.append(f"  {i}. {description} (value: {c['feature_value']}) {strength} {direction} the predicted risk.")

    lines.append("")
    if decision == "declined":
        lines.append(
            "To improve future eligibility, focus on the risk-increasing factors above — "
            "see the counterfactual 'what-if' scenario for specific, achievable changes."
        )
    else:
        lines.append("No immediate action is needed; the applicant's profile falls within acceptable risk bounds.")

    return "\n".join(lines)
