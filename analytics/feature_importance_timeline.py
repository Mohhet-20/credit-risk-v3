"""
Feature importance timeline across registered model versions.

Design decision: importances are extracted directly from each registered
model artifact (tree-based `.feature_importances_` or linear `.coef_`)
rather than requiring the training pipeline to separately persist an
importance snapshot at registration time. This means the timeline works
retroactively for every version already in the registry and stays correct
even if a future training run forgets to log importances explicitly.
"""

from __future__ import annotations

import numpy as np

from models.registry import ModelRegistry

TOP_K_DEFAULT = 10


def _unwrap_calibrated(model):
    """CalibratedClassifierCV wraps the fitted base estimator; unwrap to
    reach the object that actually exposes `feature_importances_`/`coef_`."""
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        inner = model.calibrated_classifiers_[0]
        return getattr(inner, "estimator", None) or getattr(inner, "base_estimator", None)
    return model


def _extract_importances(model, feature_names: list[str]) -> list[dict] | None:
    base = _unwrap_calibrated(model)
    if base is None:
        return None
    if hasattr(base, "feature_importances_"):
        values = np.asarray(base.feature_importances_)
    elif hasattr(base, "coef_"):
        values = np.abs(np.asarray(base.coef_)).reshape(-1)
    else:
        return None
    if len(values) != len(feature_names):
        return None
    pairs = sorted(zip(feature_names, values.tolist(), strict=False), key=lambda p: p[1], reverse=True)
    return [{"feature": f, "importance": round(v, 6)} for f, v in pairs]


def compute_feature_importance_timeline(registry: ModelRegistry, top_k: int = TOP_K_DEFAULT) -> list[dict]:
    timeline = []
    for meta in registry.list_versions():
        version = meta["version"]
        try:
            bundle = registry.load_version(version)
        except Exception:  # noqa: BLE001 - skip an unreadable version rather than fail the whole timeline
            continue
        importances = _extract_importances(bundle.model, bundle.feature_pipeline.feature_names)
        timeline.append(
            {
                "version": version,
                "model_name": meta.get("model_name"),
                "registered_at": meta.get("registered_at"),
                "top_features": importances[:top_k] if importances else None,
            }
        )
    return sorted(timeline, key=lambda t: t["registered_at"] or "")
