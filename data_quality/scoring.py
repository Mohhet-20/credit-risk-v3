"""
Composite data-quality score: a single 0-100 number a dashboard can show at
a glance, plus the component scores it's built from (validity, completeness,
uniqueness) and two metadata fields the design calls out separately
(freshness, and the validation pass/fail verdict) rather than folding into
the score itself.

Design decision: freshness is reported but deliberately NOT part of the 0-100
composite. "How stale is this dataset" and "how clean is this dataset" are
different questions -- a perfectly clean dataset that's a year old shouldn't
score lower on *quality*, it should be flagged as stale separately so a
caller can decide what to do about it (the two have different remediation
paths: staleness means "go refresh the extract," quality issues mean "go fix
the pipeline").
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from data_quality.validators import ValidationReport

_SEVERITY_PENALTY = {"critical": 15.0, "warning": 5.0, "info": 1.0}


def _validity_score(validation_report: ValidationReport) -> float:
    score = 100.0
    for issue in validation_report.issues:
        score -= _SEVERITY_PENALTY.get(issue.severity, 1.0)
    return max(0.0, round(score, 2))


def _completeness_score(profile: dict) -> float:
    pct_missing = profile.get("missing_value_summary", {}).get("pct_cells_missing", 0.0)
    return round(max(0.0, 1.0 - pct_missing) * 100, 2)


def _uniqueness_score(profile: dict) -> float:
    pct_duplicate = profile.get("duplicate_analysis", {}).get("pct_duplicate_rows", 0.0)
    return round(max(0.0, 1.0 - pct_duplicate) * 100, 2)


def _freshness_hours(dataset_path: str | Path | None) -> float | None:
    if dataset_path is None:
        return None
    path = Path(dataset_path)
    if not path.exists():
        return None
    age_seconds = time.time() - os.path.getmtime(path)
    return round(age_seconds / 3600, 2)


def compute_quality_score(
    validation_report: ValidationReport,
    profile: dict,
    dataset_path: str | Path | None = None,
) -> dict:
    validity = _validity_score(validation_report)
    completeness = _completeness_score(profile)
    uniqueness = _uniqueness_score(profile)
    # Equal-weighted average of the three dimensions that ARE quality
    # (validity/completeness/uniqueness) -- freshness is reported alongside,
    # not blended in, per the module docstring.
    overall = round((validity + completeness + uniqueness) / 3, 2)

    return {
        "overall_score": overall,
        "validity_score": validity,
        "completeness_score": completeness,
        "uniqueness_score": uniqueness,
        "freshness_hours": _freshness_hours(dataset_path),
        "validation_passed": validation_report.passed,
        "n_validation_issues": len(validation_report.issues),
    }
