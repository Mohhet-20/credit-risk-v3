"""Integration tests for the local model registry: register, promote, rollback."""

from __future__ import annotations

import pytest

from core.exceptions import ModelNotFoundError
from models.registry import ModelRegistry


def test_register_and_load_champion(model_registry: ModelRegistry, trained_model_bundle):
    champion = model_registry.load_champion()
    assert champion.metadata["model_name"] == trained_model_bundle.metadata["model_name"]
    assert model_registry.get_champion_version() == trained_model_bundle.metadata["version"]


def test_promote_new_version_updates_champion_pointer(model_registry, trained_model_bundle):
    model_registry.register_version(
        version="test-v2",
        model=trained_model_bundle.model,
        feature_pipeline=trained_model_bundle.feature_pipeline,
        metadata={**trained_model_bundle.metadata, "version": "test-v2"},
    )
    model_registry.set_champion("test-v2")
    assert model_registry.get_champion_version() == "test-v2"


def test_rollback_restores_previous_champion(model_registry, trained_model_bundle):
    model_registry.register_version(
        version="test-v2",
        model=trained_model_bundle.model,
        feature_pipeline=trained_model_bundle.feature_pipeline,
        metadata={**trained_model_bundle.metadata, "version": "test-v2"},
    )
    model_registry.set_champion("test-v2")
    assert model_registry.get_champion_version() == "test-v2"

    model_registry.rollback_to(trained_model_bundle.metadata["version"])
    assert model_registry.get_champion_version() == trained_model_bundle.metadata["version"]


def test_promote_unknown_version_raises(model_registry):
    with pytest.raises(ModelNotFoundError):
        model_registry.set_champion("does-not-exist")


def test_list_versions_returns_metadata(model_registry, trained_model_bundle):
    versions = model_registry.list_versions()
    assert any(v["version"] == trained_model_bundle.metadata["version"] for v in versions)
