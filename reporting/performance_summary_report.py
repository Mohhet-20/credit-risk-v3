"""
Performance Summary Report: prediction volume, average probability,
approximate throughput, and latency percentiles, rendered from
`MonitoringService.get_stats()` / `get_latency_stats()`.

Scoping note (documented in the rendered report too, not just here): this
platform does not yet instrument a distinct "prediction request errored"
counter (a failed `/predict` call currently surfaces as a normal HTTP error
response and a structured log line via `core/logging.py`'s correlation-ID
filter, but isn't aggregated into a rate). A true error-rate metric would
need a small counter in `ScoringService` incrementing on exception, which is
a reasonable next addition but out of scope for this pass -- reported here
as `null` rather than fabricated.
"""

from __future__ import annotations

from reporting.html_report import kpi_html, kpi_row_html, note_html, render_report


def build_performance_summary_html(stats: dict, latency_stats: dict) -> str:
    window_days = stats.get("window_days", 30)
    n_predictions = stats.get("n_predictions", 0)
    throughput_per_day = round(n_predictions / window_days, 2) if window_days else None

    kpis = kpi_row_html(
        kpi_html("Predictions", f"{n_predictions:,}"),
        kpi_html("Window", f"{window_days} days"),
        kpi_html("Throughput", f"{throughput_per_day}/day" if throughput_per_day is not None else "n/a"),
        kpi_html(
            "Avg. predicted PD",
            (
                f"{stats['avg_predicted_probability']:.1%}"
                if stats.get("avg_predicted_probability") is not None
                else "n/a"
            ),
        ),
    )

    if latency_stats.get("n", 0) == 0:
        latency_section = note_html("No latency samples recorded yet.")
    else:
        latency_kpis = kpi_row_html(
            kpi_html("p50 latency", f"{latency_stats['p50_ms']:.1f} ms"),
            kpi_html("p95 latency", f"{latency_stats['p95_ms']:.1f} ms"),
            kpi_html("p99 latency", f"{latency_stats['p99_ms']:.1f} ms"),
            kpi_html("Max latency", f"{latency_stats['max_ms']:.1f} ms"),
        )
        latency_section = latency_kpis

    error_rate_note = note_html(
        "Error-rate tracking is not yet instrumented (no per-request success/failure counter exists in "
        "<code>ScoringService</code> today) -- shown as n/a rather than a fabricated number. Correlation "
        "IDs (<code>X-Correlation-ID</code>) let you trace any individual failed request through the "
        "structured logs in the meantime.",
    )

    body = f"""
    <h2>Volume &amp; Throughput</h2>
    {kpis}

    <h2>Inference Latency</h2>
    {latency_section}

    <h2>Error Rate</h2>
    {error_rate_note}
    """
    return render_report(
        title="Performance Summary",
        subtitle=f"Rolling {window_days}-day window",
        body_html=body,
    )
