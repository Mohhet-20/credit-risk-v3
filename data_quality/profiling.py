"""
Automated data profiling: column statistics, feature distributions,
correlations, missing-value/duplicate/outlier summaries, cardinality, and a
dataset-level business summary.

Design decision: profiling is deliberately separate from validation
(`data_quality/validators.py`). Validation answers "is this data broken?"
(pass/fail, issues with severities); profiling answers "what does this data
look like?" (descriptive statistics, no verdict). They're composed together
by `data_quality/scoring.py` and `services/data_quality_service.py`, but
keeping them as two independently callable, independently testable modules
means a caller who only wants a profile (e.g., a one-off exploratory look at
a new CSV) doesn't have to run the full validation suite to get it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing.outliers import detect_outliers_iqr

HIGH_CARDINALITY_THRESHOLD = 50  # unique values, for a categorical column, above which it's flagged


def _column_statistics(df: pd.DataFrame) -> list[dict]:
    stats = []
    for col in df.columns:
        series = df[col]
        entry: dict = {
            "column": col,
            "dtype": str(series.dtype),
            "count": int(series.count()),
            "n_missing": int(series.isna().sum()),
            "pct_missing": round(float(series.isna().mean()), 4),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            described = series.describe()
            entry.update(
                {
                    "mean": _safe_round(described.get("mean")),
                    "std": _safe_round(described.get("std")),
                    "min": _safe_round(described.get("min")),
                    "p25": _safe_round(described.get("25%")),
                    "median": _safe_round(described.get("50%")),
                    "p75": _safe_round(described.get("75%")),
                    "max": _safe_round(described.get("max")),
                    "skew": _safe_round(series.skew()),
                }
            )
        else:
            top_values = series.value_counts(dropna=True).head(5)
            entry["top_values"] = [{"value": str(v), "count": int(c)} for v, c in top_values.items()]
        stats.append(entry)
    return stats


def _safe_round(value, digits: int = 4):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


def _correlations(df: pd.DataFrame, top_n: int = 15) -> list[dict]:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return []
    corr = numeric_df.corr(numeric_only=True)
    pairs = []
    columns = corr.columns.tolist()
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.isna(value):
                continue
            pairs.append({"feature_a": col_a, "feature_b": col_b, "correlation": round(float(value), 4)})
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return pairs[:top_n]


def _missing_value_summary(df: pd.DataFrame) -> dict:
    missing_pct = df.isna().mean()
    columns_with_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
    return {
        "total_missing_cells": int(df.isna().sum().sum()),
        "pct_cells_missing": round(float(df.isna().mean().mean()), 4),
        "n_columns_with_missing": int((missing_pct > 0).sum()),
        "worst_columns": [
            {"column": col, "pct_missing": round(float(pct), 4)} for col, pct in columns_with_missing.head(10).items()
        ],
    }


def _duplicate_analysis(df: pd.DataFrame) -> dict:
    n_exact_duplicates = int(df.duplicated(keep=False).sum())
    n_unique_duplicate_groups = (
        int(df[df.duplicated(keep=False)].drop_duplicates().shape[0]) if n_exact_duplicates else 0
    )
    return {
        "n_exact_duplicate_rows": n_exact_duplicates,
        "pct_duplicate_rows": round(n_exact_duplicates / max(len(df), 1), 4),
        "n_distinct_duplicate_groups": n_unique_duplicate_groups,
    }


def _outlier_summary(df: pd.DataFrame, numeric_columns: list[str] | None = None) -> list[dict]:
    columns = numeric_columns or df.select_dtypes(include=[np.number]).columns.tolist()
    return detect_outliers_iqr(df, columns).to_dict(orient="records")


def _cardinality(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in df.columns:
        n_unique = int(df[col].nunique(dropna=True))
        is_high_cardinality = not pd.api.types.is_numeric_dtype(df[col]) and n_unique > HIGH_CARDINALITY_THRESHOLD
        rows.append(
            {
                "column": col,
                "n_unique": n_unique,
                "pct_unique": round(n_unique / max(len(df), 1), 4),
                "is_high_cardinality": is_high_cardinality,
            }
        )
    return sorted(rows, key=lambda r: r["n_unique"], reverse=True)


def _business_summary(df: pd.DataFrame, target: pd.Series | None = None) -> dict:
    summary: dict = {"n_rows": len(df), "n_columns": df.shape[1]}
    if target is not None:
        rate = float(target.mean())
        summary["target_positive_rate"] = round(rate, 4)
        summary["target_class_counts"] = {str(k): int(v) for k, v in target.value_counts().items()}
    for col in ("annual_income", "loan_amount", "credit_score"):
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            summary[f"{col}_median"] = _safe_round(df[col].median())
    return summary


def generate_profile(df: pd.DataFrame, target: pd.Series | None = None) -> dict:
    """Runs the full profiling suite and returns a single JSON-serializable
    dict -- the shape persisted by `DataQualityReportRepository` and rendered
    by `reporting/data_quality_report.py`."""
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "column_statistics": _column_statistics(df),
        "correlations": _correlations(df),
        "missing_value_summary": _missing_value_summary(df),
        "duplicate_analysis": _duplicate_analysis(df),
        "outlier_summary": _outlier_summary(df),
        "cardinality": _cardinality(df),
        "business_summary": _business_summary(df, target=target),
    }
