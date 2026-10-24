"""
Small compatibility shim between XGBoost's sklearn wrapper and scikit-learn's
newer estimator-tags protocol (introduced in scikit-learn 1.6).

Context: scikit-learn 1.6+ replaced the old `_estimator_type` string
attribute with a structured `__sklearn_tags__()` API for signaling
"this is a classifier" to meta-estimators like `CalibratedClassifierCV`,
`StackingClassifier`, and `permutation_importance`. As of xgboost 2.1.x,
`XGBClassifier.__sklearn_tags__()` does not correctly populate
`estimator_type`/`classifier_tags`, so `sklearn.base.is_classifier()`
returns False for an XGBClassifier and any meta-estimator that gates on it
raises `ValueError: ... Got a regressor ...`. LightGBM and CatBoost's
wrappers are unaffected — this is XGBoost-specific.

This module patches `XGBClassifier.__sklearn_tags__` at import time to
correctly report classifier tags. Importing this module (for its side
effect) anywhere XGBClassifier is combined with scikit-learn meta-estimators
is sufficient; the patch is idempotent and safe to apply multiple times.
"""

from __future__ import annotations

from sklearn.utils import ClassifierTags
from xgboost import XGBClassifier

if not getattr(XGBClassifier, "_credit_risk_platform_tags_patched", False):
    _original_sklearn_tags = XGBClassifier.__sklearn_tags__

    def _patched_sklearn_tags(self):
        tags = _original_sklearn_tags(self)
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        return tags

    XGBClassifier.__sklearn_tags__ = _patched_sklearn_tags
    XGBClassifier._credit_risk_platform_tags_patched = True
