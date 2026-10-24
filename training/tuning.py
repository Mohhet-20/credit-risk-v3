"""
Optuna-driven hyperparameter search for each model family in the comparison.

Design decision: Optuna (TPE sampler) rather than GridSearchCV — grid search
scales combinatorially and wastes budget on clearly-bad regions of the search
space, while Optuna's Bayesian-ish sampler concentrates trials where the
objective is improving. We optimize ROC-AUC under stratified CV (cheaper
`tuning_cv_folds`, typically 3) inside the objective, then re-validate the
winning config with the full `n_cv_folds` (5) elsewhere for reporting. A
`MedianPruner` stops clearly-bad trials early to conserve compute — material
on a 2-vCPU box.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import core.sklearn_compat  # noqa: F401  (side-effect: patches XGBClassifier sklearn tags)
from core.logging import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_BUILDERS: dict[str, Callable[..., Any]] = {
    "logistic_regression": lambda **p: LogisticRegression(max_iter=2000, **p),
    "random_forest": lambda **p: RandomForestClassifier(n_jobs=-1, random_state=42, **p),
    "xgboost": lambda **p: XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=42, tree_method="hist", **p),
    "lightgbm": lambda **p: LGBMClassifier(n_jobs=-1, random_state=42, verbosity=-1, **p),
    "catboost": lambda **p: CatBoostClassifier(random_state=42, verbose=False, **p),
}


# Business-logic monotonicity: +1 = PD may only rise with the feature, -1 =
# PD may only fall. Enforced on the gradient-boosted families so the model
# can never learn e.g. "a higher credit score raises risk" from noise in a
# sparse region -- a behaviour a credit committee would reject on sight and
# one that would produce indefensible adverse-action reasons.
MONOTONE_DIRECTIONS: dict[str, int] = {
    "credit_score": -1,
    "annual_income": -1,
    "employment_length_years": -1,
    "is_stable_employment": -1,
    "months_since_last_delinquency": -1,
    "savings_balance": -1,
    "total_liquid_assets": -1,
    "liquidity_to_loan_ratio": -1,
    "debt_to_income_ratio": 1,
    "revolving_utilization": 1,
    "num_delinquencies_2yrs": 1,
    "num_derogatory_marks": 1,
    "num_credit_inquiries_6m": 1,
    "loan_to_income_ratio": 1,
    "payment_to_income_ratio": 1,
    "delinquency_severity": 1,
    "has_recent_delinquency": 1,
    "interest_rate": 1,
    "utilization_x_low_score": 1,
}
MONOTONE_MODELS = ("xgboost", "lightgbm", "catboost")


def _monotone_kwargs(model_name: str, feature_names: list[str] | None) -> dict[str, Any]:
    if model_name not in MONOTONE_MODELS or not feature_names:
        return {}
    constraints = [MONOTONE_DIRECTIONS.get(f, 0) for f in feature_names]
    if model_name == "lightgbm":
        return {"monotone_constraints": constraints}
    # XGBoost and CatBoost both take the "(1,0,-1)" string form; CatBoost
    # also rewrites a list argument in its constructor, which breaks
    # sklearn's clone() inside StackingClassifier.
    return {"monotone_constraints": "(" + ",".join(str(c) for c in constraints) + ")"}


def _class_weight_kwargs(model_name: str, y: pd.Series) -> dict[str, Any]:
    """Per-model-family kwargs implementing `configs/model_training.yaml`'s
    `class_imbalance.strategy: class_weight`. Every model family exposes
    imbalance handling under a different constructor argument (sklearn/
    LightGBM: `class_weight`; CatBoost: `auto_class_weights`; XGBoost has
    neither and instead wants `scale_pos_weight`, the negative:positive
    ratio), so this can't be a single shared kwarg passed to all builders."""
    if model_name in ("logistic_regression", "random_forest", "lightgbm"):
        return {"class_weight": "balanced"}
    if model_name == "catboost":
        return {"auto_class_weights": "Balanced"}
    if model_name == "xgboost":
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        return {"scale_pos_weight": n_neg / n_pos} if n_pos > 0 else {}
    raise ValueError(f"No class-weight strategy defined for model {model_name!r}")


def build_model(
    model_name: str,
    params: dict[str, Any],
    *,
    y: pd.Series | None = None,
    class_imbalance_strategy: str | None = None,
    feature_names: list[str] | None = None,
) -> Any:
    """Builds a model from `MODEL_BUILDERS`, optionally merging in class-
    imbalance handling per `class_imbalance.strategy` -- previously this
    config key was declared in `configs/model_training.yaml` but never
    referenced anywhere in training code (dead config on a dataset with a
    real minority-class imbalance)."""
    extra: dict[str, Any] = {}
    if class_imbalance_strategy in ("class_weight", "both"):
        if y is None:
            raise ValueError("class_imbalance strategy 'class_weight'/'both' requires `y`.")
        extra = _class_weight_kwargs(model_name, y)
    if class_imbalance_strategy in ("smote", "both"):
        raise NotImplementedError(
            "class_imbalance.strategy 'smote' is not implemented -- no SMOTE dependency "
            "(e.g. imbalanced-learn) is installed in this project. Use 'class_weight' or 'none'."
        )
    elif class_imbalance_strategy not in (None, "none", "class_weight"):
        raise ValueError(f"Unknown class_imbalance.strategy: {class_imbalance_strategy!r}")
    extra.update(_monotone_kwargs(model_name, feature_names))
    return MODEL_BUILDERS[model_name](**{**extra, **params})


def _suggest_params(trial: optuna.Trial, search_space: dict[str, dict]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        ptype = spec["type"]
        if ptype == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif ptype == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search space type: {ptype}")
    return params


def tune_model(
    model_name: str,
    search_space: dict[str, dict],
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 15,
    cv_folds: int = 3,
    timeout: int | None = 300,
    sample_weight: pd.Series | None = None,
    random_state: int = 42,
    class_imbalance_strategy: str | None = None,
) -> tuple[dict, float]:
    """Runs an Optuna study for one model family; returns (best_params, best_cv_roc_auc)."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, search_space)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = build_model(
                model_name,
                params,
                y=y_train,
                class_imbalance_strategy=class_imbalance_strategy,
                feature_names=list(X.columns),
            )
            if sample_weight is not None:
                model.fit(X_train, y_train, sample_weight=sample_weight.iloc[train_idx])
            else:
                model.fit(X_train, y_train)
            proba = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, proba))
        return float(np.mean(scores))

    sampler = TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    logger.info(
        f"Tuning complete for {model_name}",
        extra={"context": {"best_value": round(study.best_value, 4), "best_params": study.best_params}},
    )
    return study.best_params, study.best_value
