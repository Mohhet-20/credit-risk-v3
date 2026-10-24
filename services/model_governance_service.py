"""
Model governance business logic: list/compare registered versions, promote a
version to champion, roll back, and invalidate the in-process model cache.

Design decision: `bust_cache_fn` is injected as a plain callable rather than
this service importing `api.dependencies.bust_model_cache` directly -- the
service layer must not depend on the API layer (that dependency direction
would point the wrong way in a layered architecture and create a circular
import, since `api/dependencies.py` is what constructs this service). The
router supplies the concrete callback; the service only needs "a function
that invalidates the cache," not where that cache lives.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from configs.settings import get_settings
from core.cache import get_cache
from core.exceptions import ModelNotFoundError
from models.registry import ModelBundle, ModelRegistry
from repositories.audit_log_repository import AuditLogRepository
from repositories.model_version_repository import ModelVersionRepository

MODEL_COMPARISON_CACHE_KEY = "model_governance:comparison"
MODEL_COMPARISON_CACHE_TTL_SECONDS = 60.0


class ModelGovernanceService:
    def __init__(
        self,
        registry: ModelRegistry,
        model_version_repo: ModelVersionRepository,
        audit_repo: AuditLogRepository,
        bust_cache_fn: Callable[[], None],
    ):
        self.registry = registry
        self.model_version_repo = model_version_repo
        self.audit_repo = audit_repo
        self.bust_cache_fn = bust_cache_fn

    def list_versions(self) -> list[dict]:
        champion = self.registry.get_champion_version()
        return [{**v, "is_champion": v["version"] == champion} for v in self.registry.list_versions()]

    @staticmethod
    def get_comparison() -> list[dict]:
        """Cached for `MODEL_COMPARISON_CACHE_TTL_SECONDS`: this table only
        changes after a full retrain (`make train`), so re-reading the CSV
        from disk on every dashboard poll is pure waste. A short TTL (rather
        than an explicit invalidation hook, as used for the model bundle
        cache on promote/rollback) is the right tradeoff here -- retraining
        is an offline, human-initiated event, not something callers need to
        see reflected within milliseconds."""
        cache = get_cache()
        cached = cache.get(MODEL_COMPARISON_CACHE_KEY)
        if cached is not None:
            return cached

        settings = get_settings()
        comparison_path = settings.data_processed_dir / "checkpoints" / "model_comparison.csv"
        if not comparison_path.exists():
            raise ModelNotFoundError("No model comparison table has been generated yet.")
        result = pd.read_csv(comparison_path).to_dict(orient="records")
        cache.set(MODEL_COMPARISON_CACHE_KEY, result, MODEL_COMPARISON_CACHE_TTL_SECONDS)
        return result

    def promote(self, version: str, *, actor_user_id: str) -> ModelBundle:
        self.registry.set_champion(version)  # raises ModelNotFoundError for unknown versions
        self.bust_cache_fn()
        bundle = self.registry.load_version(version)
        self.model_version_repo.upsert(
            version=version,
            model_name=bundle.metadata["model_name"],
            stage="Production",
            metrics=bundle.metadata.get("test_metrics"),
            decision_threshold=bundle.metadata.get("decision_threshold"),
        )
        self.audit_repo.log_action(user_id=actor_user_id, action="promote_model", resource=f"model_version:{version}")
        return bundle

    def rollback(self, version: str, *, actor_user_id: str) -> ModelBundle:
        self.registry.rollback_to(version)  # raises ModelNotFoundError for unknown versions
        self.bust_cache_fn()
        bundle = self.registry.load_version(version)
        self.audit_repo.log_action(user_id=actor_user_id, action="rollback_model", resource=f"model_version:{version}")
        return bundle

    def reload_cache(self) -> None:
        self.bust_cache_fn()
