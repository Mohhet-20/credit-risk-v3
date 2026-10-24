"""
The canonical Feature Pipeline — the single artifact shared by training and
serving.

Design decision (train/serve skew prevention): the #1 cause of production ML
bugs is a mismatch between how features were computed during training vs at
inference time. Rather than re-implementing preprocessing in the API layer,
`FeaturePipeline` is fit ONCE during training, serialized alongside the model
(joblib), and loaded as-is by the FastAPI service. `raw -> engineered`
is therefore byte-for-byte identical in both paths by construction.

Protected/proxy attributes (`MODEL_EXCLUDED_FEATURES`) are dropped before
any stage runs, so no fitted statistic -- not even an imputation median --
is ever computed from them.

Stages: MissingValueHandler -> OutlierWinsorizer -> CreditFeatureEngineer ->
KFoldTargetEncoder(high-cardinality) -> OneHotFrameEncoder(low-cardinality) ->
FeatureSelector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from core.constants import MODEL_CATEGORICAL_FEATURES, MODEL_EXCLUDED_FEATURES, MODEL_NUMERIC_FEATURES
from core.exceptions import FeaturePipelineError
from core.logging import get_logger
from preprocessing.feature_engineering import CreditFeatureEngineer
from preprocessing.feature_selection import FeatureSelector
from preprocessing.missing_values import MissingValueHandler
from preprocessing.outliers import OutlierWinsorizer
from preprocessing.target_encoding import KFoldTargetEncoder, OneHotFrameEncoder

logger = get_logger(__name__)

HIGH_CARDINALITY_COLUMNS = ["loan_purpose"]
LOW_CARDINALITY_COLUMNS = ["home_ownership", "education_level", "employment_status"]
# Descriptive tiers (catalog/reporting only). Not model inputs: they are
# unconstrained step-function copies of `credit_score` / `revolving_utilization`,
# so feeding them in would let the model route around the monotonic
# constraints on those features (see training/tuning.py::MONOTONE_DIRECTIONS).
DESCRIPTIVE_BAND_COLUMNS = ["credit_utilization_band", "credit_score_band"]
NON_FEATURE_COLUMNS = ["customer_id", "application_id", "application_date", *DESCRIPTIVE_BAND_COLUMNS]


def _drop_excluded(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=[c for c in MODEL_EXCLUDED_FEATURES if c in X.columns])


@dataclass
class FeaturePipeline:
    apply_feature_selection: bool = True
    corr_threshold: float = 0.95
    importance_top_k: int | None = None
    random_state: int = 42

    _is_fitted: bool = field(default=False, init=False)

    def fit(self, X_raw: pd.DataFrame, y: pd.Series) -> FeaturePipeline:
        logger.info("Fitting FeaturePipeline", extra={"context": {"n_rows": len(X_raw)}})
        X_raw = _drop_excluded(X_raw)

        self.missing_handler = MissingValueHandler(
            numeric_columns=list(MODEL_NUMERIC_FEATURES),
            categorical_columns=list(MODEL_CATEGORICAL_FEATURES),
        )
        X = self.missing_handler.fit(X_raw).transform(X_raw)

        self.outlier_handler = OutlierWinsorizer(columns=list(MODEL_NUMERIC_FEATURES))
        X = self.outlier_handler.fit(X).transform(X)

        self.feature_engineer = CreditFeatureEngineer()
        X = self.feature_engineer.fit(X).transform(X)

        self.target_encoder = KFoldTargetEncoder(columns=HIGH_CARDINALITY_COLUMNS, random_state=self.random_state)
        X = self.target_encoder.fit_transform_oof(X, y)

        onehot_cols = [c for c in LOW_CARDINALITY_COLUMNS if c in X.columns]
        self.onehot_encoder = OneHotFrameEncoder(columns=onehot_cols)
        X = self.onehot_encoder.fit(X).transform(X)

        drop_cols = [c for c in NON_FEATURE_COLUMNS if c in X.columns]
        X = X.drop(columns=drop_cols)

        if self.apply_feature_selection:
            self.feature_selector = FeatureSelector(
                corr_threshold=self.corr_threshold,
                importance_top_k=self.importance_top_k,
                random_state=self.random_state,
            )
            X = self.feature_selector.fit_transform(X, y)
        else:
            self.feature_selector = None

        self.final_feature_names_ = X.columns.tolist()
        self._is_fitted = True
        logger.info(
            "FeaturePipeline fit complete",
            extra={"context": {"n_final_features": len(self.final_feature_names_)}},
        )
        return self

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise FeaturePipelineError("FeaturePipeline.transform() called before fit().")

        X = self.missing_handler.transform(_drop_excluded(X_raw))
        X = self.outlier_handler.transform(X)
        X = self.feature_engineer.transform(X)
        X = self.target_encoder.transform(X)
        X = self.onehot_encoder.transform(X)

        drop_cols = [c for c in NON_FEATURE_COLUMNS if c in X.columns]
        X = X.drop(columns=drop_cols)

        for col in self.final_feature_names_:
            if col not in X.columns:
                X[col] = 0
        X = X[self.final_feature_names_]
        return X

    def fit_transform(self, X_raw: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X_raw, y).transform(X_raw)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("FeaturePipeline saved", extra={"context": {"path": str(path)}})

    @staticmethod
    def load(path: str | Path) -> FeaturePipeline:
        return joblib.load(path)

    @property
    def feature_names(self) -> list[str]:
        return list(self.final_feature_names_)
