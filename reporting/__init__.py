"""
Self-contained HTML report generators: Model Comparison, Evaluation, Drift,
Performance Summary, and per-applicant Explainability reports.

Design decision: HTML rather than a second PDF pipeline. The platform
already has a reportlab-based PDF report generator
(`explainability/report_generator.py`) for the per-applicant
adverse-action-notice workflow, which is unchanged and still available via
`/explain/report`. These five report types are read on a screen (a risk
analyst reviewing a model comparison, a data scientist checking a drift
report), not printed or mailed, so a self-contained HTML document (inline
CSS, no external assets, charts embedded as base64 PNGs) is the simpler and
more appropriate choice -- no new heavyweight dependency, opens in any
browser, and easy to iterate on formatting without reportlab's flowable API.
"""
