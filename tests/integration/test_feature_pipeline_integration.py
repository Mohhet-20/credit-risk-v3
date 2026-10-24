"""Integration test: the full FeaturePipeline against realistic synthetic data,
including a save/load round trip (the exact serialization path used by the
model registry and the API's `get_model_bundle` dependency)."""

from __future__ import annotations

import numpy as np

from feature_store.pipeline import FeaturePipeline


def test_feature_pipeline_produces_fully_numeric_frame_with_no_nans(fitted_feature_pipeline, raw_dataset_no_leak):
    X, y = raw_dataset_no_leak
    transformed = fitted_feature_pipeline.transform(X)

    assert transformed.isna().sum().sum() == 0
    non_numeric = transformed.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    assert non_numeric == []


def test_feature_pipeline_transform_is_deterministic(fitted_feature_pipeline, raw_dataset_no_leak):
    X, _ = raw_dataset_no_leak
    first = fitted_feature_pipeline.transform(X.head(20))
    second = fitted_feature_pipeline.transform(X.head(20))
    assert np.allclose(
        first.select_dtypes(include="number").to_numpy(), second.select_dtypes(include="number").to_numpy()
    )


def test_feature_pipeline_handles_unseen_category_gracefully(fitted_feature_pipeline, raw_dataset_no_leak):
    X, _ = raw_dataset_no_leak
    row = X.head(1).copy()
    row["state"] = "ZZ"  # never seen during fit
    transformed = fitted_feature_pipeline.transform(row)
    assert transformed.isna().sum().sum() == 0


def test_feature_pipeline_save_load_round_trip(fitted_feature_pipeline, raw_dataset_no_leak, tmp_path):
    X, _ = raw_dataset_no_leak
    save_path = tmp_path / "feature_pipeline.joblib"
    fitted_feature_pipeline.save(save_path)

    reloaded = FeaturePipeline.load(save_path)
    original_output = fitted_feature_pipeline.transform(X.head(10))
    reloaded_output = reloaded.transform(X.head(10))

    assert list(original_output.columns) == list(reloaded_output.columns)
    assert np.allclose(
        original_output.select_dtypes(include="number").to_numpy(),
        reloaded_output.select_dtypes(include="number").to_numpy(),
    )
