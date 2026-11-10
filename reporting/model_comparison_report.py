"""
Model Comparison Report: cross-model metrics (from the training pipeline's
`model_comparison.csv`, surfaced today via `ModelGovernanceService.get_comparison()`)
plus the registered model-version list, rendered as one HTML document instead
of two separate API responses a caller has to reconcile by hand.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reporting.html_report import df_to_html_table, fig_to_base64_img, note_html, render_report


def _pr_auc_bar_fig(comparison_rows: list[dict]):
    df = pd.DataFrame(comparison_rows)
    if "pr_auc" not in df.columns or df.empty:
        return None
    df = df.sort_values("pr_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.5 * len(df))))
    colors = ["#2c7fb8" if not df.iloc[i].get("is_best", False) else "#e07b39" for i in range(len(df))]
    ax.barh(df["model"], df["pr_auc"], color=colors)
    ax.set_xlabel("PR-AUC")
    ax.set_title("Model comparison (higher is better)")
    fig.tight_layout()
    return fig


def build_model_comparison_report_html(
    comparison_rows: list[dict],
    versions: list[dict],
    champion_version: str | None,
) -> str:
    body_parts = []

    if not comparison_rows:
        body_parts.append(
            note_html(
                "No model comparison table has been generated yet -- run the training pipeline "
                "(<code>make train</code>) to produce one.",
            )
        )
    else:
        best = max(comparison_rows, key=lambda r: r.get("pr_auc", 0.0))
        for row in comparison_rows:
            row["is_best"] = row.get("model") == best.get("model")
        fig = _pr_auc_bar_fig(comparison_rows)
        chart_html = fig_to_base64_img(fig, "Model comparison chart") if fig is not None else ""
        table_html = df_to_html_table(pd.DataFrame(comparison_rows).drop(columns=["is_best"], errors="ignore"))
        body_parts.append(
            f"""
        <h2>Cross-Model Metrics</h2>
        {table_html}
        {chart_html}
        <p>Best candidate by PR-AUC: <b>{best.get('model')}</b> ({best.get('pr_auc')}).</p>
        """
        )

    if versions:
        versions_df = pd.DataFrame(
            [
                {
                    "version": v.get("version"),
                    "model_name": v.get("model_name"),
                    "registered_at": v.get("registered_at"),
                    "decision_threshold": v.get("decision_threshold"),
                    "champion": "★" if v.get("is_champion") else "",
                }
                for v in versions
            ]
        )
        body_parts.append(
            f"""
        <h2>Registered Versions</h2>
        {df_to_html_table(versions_df)}
        """
        )
    else:
        body_parts.append(note_html("No model versions are registered yet."))

    return render_report(
        title="Model Comparison Report",
        subtitle=f"Current champion: {champion_version or 'none registered'}",
        body_html="".join(body_parts),
    )
