"""
Downloadable explainability reports (PDF).

Design decision: reports are generated with matplotlib (for SHAP/PDP charts,
rendered to PNG buffers — no display backend needed, safe in a headless
container) and assembled with reportlab's flowable API rather than an HTML
templating + headless-browser-to-PDF pipeline (e.g. WeasyPrint via wkhtmltopdf),
which would add a heavy system dependency for a container that otherwise only
needs Python packages. Two report types are supported: a per-applicant report
(local SHAP waterfall + plain-English explanation + counterfactual scenario)
for a loan officer / adverse-action-notice workflow, and a global model report
(feature importance + calibration curve + model comparison table) for model
governance / risk-committee review.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core.logging import get_logger

logger = get_logger(__name__)


def _fig_to_image_flowable(fig, width_inches: float = 6.0) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_inches * inch, height=width_inches * 0.55 * inch)


def _waterfall_figure(contributions: list[dict], base_value: float, top_n: int = 10):
    top = contributions[:top_n]
    features = [c["feature"] for c in top][::-1]
    values = [c["shap_value"] for c in top][::-1]
    colors_list = ["#d9534f" if v > 0 else "#5cb85c" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(features, values, color=colors_list)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution (impact on predicted default probability)")
    ax.set_title(f"Top factors (base rate: {base_value:.1%})")
    fig.tight_layout()
    return fig


def generate_applicant_report(
    output_path: str | Path,
    applicant_id: str,
    predicted_probability: float,
    decision_threshold: float,
    local_explanation: dict,
    plain_english_text: str,
    counterfactual_changes: list[dict] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=18)
    doc = SimpleDocTemplate(str(output_path), pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = []

    story.append(Paragraph("Credit Risk Explainability Report", title_style))
    story.append(Paragraph(f"Applicant: {applicant_id}", styles["Normal"]))
    decision = "DECLINED" if predicted_probability >= decision_threshold else "APPROVED"
    story.append(
        Paragraph(
            f"Decision: <b>{decision}</b> &nbsp;|&nbsp; Predicted default probability: "
            f"<b>{predicted_probability:.1%}</b> &nbsp;|&nbsp; Threshold: {decision_threshold:.1%}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.25 * inch))

    fig = _waterfall_figure(local_explanation["contributions"], local_explanation["base_value"])
    story.append(_fig_to_image_flowable(fig))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Plain-English Explanation", styles["Heading2"]))
    for line in plain_english_text.split("\n"):
        story.append(Paragraph(line if line.strip() else "&nbsp;", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    if counterfactual_changes:
        story.append(Paragraph("What-If Scenario (Path to Approval)", styles["Heading2"]))
        table_data = [["Feature", "From", "To", "Resulting Risk"]]
        for c in counterfactual_changes:
            table_data.append(
                [
                    c["feature"],
                    f"{c['from']:.3g}",
                    f"{c['to']:.3g}",
                    f"{c['resulting_probability']:.1%}",
                ]
            )
        table = Table(table_data, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                ]
            )
        )
        story.append(table)

    doc.build(story)
    logger.info("Applicant report generated", extra={"context": {"path": str(output_path)}})
    return output_path


def generate_global_report(
    output_path: str | Path,
    comparison_df: pd.DataFrame,
    global_importance_df: pd.DataFrame,
    champion_model_name: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output_path), pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = [
        Paragraph("Credit Risk Model — Governance Report", styles["Title"]),
        Paragraph(f"Champion model: <b>{champion_model_name}</b>", styles["Normal"]),
        Spacer(1, 0.25 * inch),
    ]

    story.append(Paragraph("Model Comparison", styles["Heading2"]))
    comp_cols = ["model", "roc_auc", "pr_auc", "brier_score", "f1"]
    comp_cols = [c for c in comp_cols if c in comparison_df.columns]
    table_data = [comp_cols] + comparison_df[comp_cols].round(4).astype(str).values.tolist()
    table = Table(table_data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Global Feature Importance (mean |SHAP|)", styles["Heading2"]))
    top_features = global_importance_df.head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top_features["feature"][::-1], top_features["mean_abs_shap"][::-1], color="#2c7fb8")
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    story.append(_fig_to_image_flowable(fig))

    doc.build(story)
    logger.info("Global governance report generated", extra={"context": {"path": str(output_path)}})
    return output_path
