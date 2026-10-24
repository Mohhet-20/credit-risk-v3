"""
Per-model-version SHAP explainer cache.

Building an explainer means sampling a background set, running it through
the feature pipeline and constructing the SHAP backend -- far too slow to do
on every request, and the result only changes when the champion changes.
Explainers are therefore cached by model version; a promotion or rollback
produces a new version string and so a fresh explainer on first use.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import pandas as pd

from configs.settings import get_settings
from core.constants import TARGET_COLUMN
from explainability.explanations import ModelExplainer
from models.registry import ModelBundle

_lock = threading.Lock()
_explainers: dict[str, ModelExplainer] = {}
_MAX_CACHED_VERSIONS = 4


BACKGROUND_FILENAME = "background_sample.csv"


@lru_cache(maxsize=_MAX_CACHED_VERSIONS)
def background_sample(version: str, n: int = 100) -> pd.DataFrame:
    """Raw-schema background rows for `version`. Training writes a sample of
    the model's own training window next to the artifact, so serving never
    depends on the raw data lake being mounted; the raw CSV is a fallback for
    versions registered before that file existed."""
    settings = get_settings()
    path = settings.model_artifact_dir / version / BACKGROUND_FILENAME
    if not path.exists():
        path = settings.data_raw_dir / "credit_applications.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    return df.sample(min(len(df), n), random_state=42)


def get_explainer(bundle: ModelBundle) -> ModelExplainer:
    version = bundle.metadata["version"]
    with _lock:
        cached = _explainers.get(version)
    if cached is not None:
        return cached

    background_t = bundle.feature_pipeline.transform(background_sample(version))
    explainer = ModelExplainer(model=bundle.model, feature_names=bundle.feature_pipeline.feature_names)
    explainer.build(background_t)

    with _lock:
        if len(_explainers) >= _MAX_CACHED_VERSIONS:
            _explainers.pop(next(iter(_explainers)))
        _explainers[version] = explainer
    return explainer


def clear() -> None:
    with _lock:
        _explainers.clear()
    background_sample.cache_clear()
