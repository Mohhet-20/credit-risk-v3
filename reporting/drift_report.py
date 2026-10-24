"""
Drift Report: HTML rendering of `MonitoringService.get_drift()`'s
PSI-based feature-drift output (`data_quality/drift_detection.py`).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reporting.html_report import df_to_html_table, fig_to_base64_img, kpi_html, kpi_row_html, note_html, render_report

_SEVERITY_COLOR = {"significant": "#c0392b", "moderate": "#e07b39", "none": "#2c7fb8"}


def _psi_bar_fig(feature_drift: list[dict]):
    if not feature_drift:
        return None
    df = pd.DataFrame(feature_drift).sort_values("psi", ascending=True)
    colors = [_SEVERITY_COLOR.get(s, "#2c7fb8") for s in df["severity"]]
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.35 * len(df))))
    ax.barh(df["feature"], df["psi"], color=colors)
    ax.axvline(0.1, linestyle="--", color="grey", linewidth=0.8, label="moderate (0.10)")
    ax.axvline(0.25, linestyle="--", color="black", linewidth=0.8, label="significant (0.25)")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Feature drift (reference vs. current window)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def build_drift_report_html(drift_result: dict) -> str:
    if "error" in drift_result:
        body = note_html(drift_result["error"])
        return render_report(title="Drift Report", subtitle="Feature/data drift monitoring", body_html=body)

    severity = drift_result.get("overall_severity", "none")
    feature_drift = drift_result.get("feature_drift", [])
    n_significant = sum(1 for f in feature_drift if f.get("severity") == "significant")
    n_moderate = sum(1 for f in feature_drift if f.get("severity") == "moderate")

    kpis = kpi_row_html(
        kpi_html("Overall severity", severity.upper()),
        kpi_html("Features monitored", str(len(feature_drift))),
        kpi_html("Significant drift", str(n_significant)),
        kpi_html("Moderate drift", str(n_moderate)),
    )

    fig = _psi_bar_fig(feature_drift)
    chart_html = fig_to_base64_img(fig, "PSI by feature") if fig is not None else ""
    table_html = df_to_html_table(pd.DataFrame(feature_drift)) if feature_drift else note_html("No features scored.")

    note = ""
    if severity == "significant":
        note = note_html(
            "One or more features show <b>significant</b> drift (PSI &ge; 0.25). Consider retraining or "
            "investigating the upstream data source before continuing to serve the current champion model.",
        )
    elif severity == "moderate":
        note = note_html(
            "One or more features show <b>moderate</b> drift (0.10 &le; PSI &lt; 0.25). Worth monitoring; "
            "not yet a retraining trigger on its own.",
        )

    body = f"""
    <h2>Summary</h2>
    {kpis}
    {note}

    <h2>Feature Drift (PSI)</h2>
    {chart_html}
    {table_html}
    """
    return render_report(title="Drift Report", subtitle="Feature/data drift monitoring (PSI)", body_html=body)
