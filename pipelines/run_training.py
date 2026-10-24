"""
End-to-end training pipeline orchestrator.

Design decision: the pipeline is exposed as a staged CLI (`--stage prepare`,
`--stage tune --model xgboost`, `--stage train_final`, `--stage ensemble`,
`--stage finalize`) rather than one monolithic `run()` call. On constrained
CI/dev hardware, this lets each stage run as an independent, resumable
process — state is checkpointed to disk between stages via joblib — which
also mirrors how a real training pipeline gets orchestrated as discrete
Airflow/Dagster/Step-Functions tasks rather than a single long-lived script.
`--stage all` runs everything sequentially for convenience on bigger hardware.

Validation design (what a model-risk reviewer checks first):
  - Out-of-time split: applications are ordered by `application_date`; the
    oldest 60% train, the next 20% is the validation window (calibration,
    model selection, threshold choice), and the newest 20% is an untouched
    out-of-time (OOT) test window. A random split would let the model "see
    the future" and overstate how it performs on next quarter's applicants.
  - Nothing is tuned on the OOT window: threshold and champion are chosen on
    validation, and the OOT numbers are reported once, at the end.
  - Duplicate applications are removed before splitting so the same loan can
    never sit on both sides of the train/test boundary.
  - Parsimony: an ensemble only displaces the best single model if it beats
    it by `ensemble.min_improvement` on the selection metric (SR 11-7 favours
    the simplest model that does the job; a single GBM is also exactly
    explainable with TreeSHAP for adverse-action reasons).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import yaml

import core.sklearn_compat  # noqa: F401  (side-effect: patches XGBClassifier sklearn tags)
from analytics.fairness import disparate_impact_report
from configs.settings import get_settings
from core.constants import TARGET_COLUMN
from core.logging import configure_logging, get_logger
from data_quality.drift_detection import compute_numeric_psi
from explainability import explainer_cache
from feature_store.pipeline import FeaturePipeline
from mlflow_tracking.tracker import ExperimentTracker
from models.registry import ModelRegistry
from training.calibration import calibrate_prefit_model, calibration_metrics
from training.credit_metrics import credit_validation_metrics, decile_table
from training.cross_validation import run_stratified_cv
from training.ensembles import BlendingEnsemble, build_stacking_ensemble
from training.model_comparison import build_comparison_table, evaluate_candidate
from training.threshold_optimization import evaluate_at_threshold, optimize_threshold
from training.tuning import build_model, tune_model

warnings.filterwarnings("ignore")
configure_logging()
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "processed" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = CHECKPOINT_DIR / "state.joblib"
PARAMS_PATH = CHECKPOINT_DIR / "tuned_params.json"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_training.yaml"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]


def load_config() -> dict:
    with open(MODEL_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_state() -> dict:
    return joblib.load(STATE_PATH) if STATE_PATH.exists() else {}


def _save_state(state: dict) -> None:
    joblib.dump(state, STATE_PATH)


def _load_params() -> dict:
    return json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else {}


def _save_params(params: dict) -> None:
    PARAMS_PATH.write_text(json.dumps(params, indent=2, default=str))


def _tracker() -> ExperimentTracker:
    settings = get_settings()
    return ExperimentTracker(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)


def _numeric_metrics(row: dict) -> dict:
    return {k: v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _out_of_time_split(df: pd.DataFrame, train_frac: float = 0.6, valid_frac: float = 0.2):
    df = df.sort_values("application_date", kind="mergesort").reset_index(drop=True)
    n = len(df)
    i_valid, i_test = int(n * train_frac), int(n * (train_frac + valid_frac))
    return df.iloc[:i_valid], df.iloc[i_valid:i_test], df.iloc[i_test:]


def _window(df: pd.DataFrame) -> dict:
    return {
        "start": str(df["application_date"].min().date()),
        "end": str(df["application_date"].max().date()),
        "n": int(len(df)),
        "default_rate": round(float(df[TARGET_COLUMN].mean()), 4),
    }


def stage_prepare(data_path: str) -> None:
    cfg = load_config()
    df = pd.read_csv(data_path, parse_dates=["application_date"])
    n_raw = len(df)
    df = df.drop_duplicates(subset="application_id", keep="first")

    train_df, valid_df, test_df = _out_of_time_split(df)
    split = {
        "train": _window(train_df),
        "validation": _window(valid_df),
        "oot_test": _window(test_df),
        "duplicates_removed": n_raw - len(df),
    }

    def xy(frame: pd.DataFrame):
        return frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN].reset_index(drop=True)

    X_train, y_train = xy(train_df)
    X_calib, y_calib = xy(valid_df)
    X_test, y_test = xy(test_df)

    fp = FeaturePipeline(importance_top_k=None, random_state=cfg["random_state"])
    Xt_train = fp.fit_transform(X_train, y_train)
    Xt_calib = fp.transform(X_calib)
    Xt_test = fp.transform(X_test)

    # "calib" is the validation window: calibration, model selection and
    # threshold choice all happen here; "test" is the out-of-time window.
    state = {
        "feature_pipeline": fp,
        "split": split,
        "Xt_train": Xt_train,
        "y_train": y_train,
        "Xt_calib": Xt_calib,
        "y_calib": y_calib,
        "Xt_test": Xt_test,
        "y_test": y_test,
        "X_test_raw": X_test.reset_index(drop=True),
        "background_raw": X_train.sample(min(len(X_train), 500), random_state=cfg["random_state"]),
        "fitted_models": {},
        "calibrated_models": {},
        "cv_results": {},
        "comparison_rows": [],
        "validation_rows": [],
    }
    _save_state(state)
    logger.info("Stage `prepare` complete", extra={"context": {**split, "n_features": Xt_train.shape[1]}})


def stage_tune(model_name: str) -> None:
    cfg = load_config()
    state = _load_state()
    Xt_train, y_train = state["Xt_train"], state["y_train"]

    model_cfg = cfg["models"][model_name]
    if not model_cfg.get("enabled", True):
        logger.info(f"Model {model_name} disabled in config, skipping tuning.")
        return

    best_params, best_score = tune_model(
        model_name=model_name,
        search_space=model_cfg["search_space"],
        X=Xt_train,
        y=y_train,
        n_trials=cfg["optuna_n_trials"],
        cv_folds=cfg["tuning_cv_folds"],
        timeout=cfg["optuna_timeout_seconds"],
        random_state=cfg["random_state"],
        class_imbalance_strategy=cfg["class_imbalance"]["strategy"],
    )
    params = _load_params()
    params[model_name] = {"best_params": best_params, "cv_roc_auc": best_score}
    _save_params(params)

    tracker = _tracker()
    with tracker.start_run(run_name=f"tune_{model_name}", tags={"stage": "tune", "model": model_name}):
        tracker.log_params({"model": model_name, **best_params})
        tracker.log_metrics({"cv_roc_auc": best_score})

    logger.info(f"Stage `tune` complete for {model_name}", extra={"context": params[model_name]})


def _record_candidate(state: dict, name: str, model, Xt_valid, y_valid, Xt_test, y_test) -> None:
    """Scores a candidate on both windows: validation metrics drive selection,
    OOT test metrics are what the comparison table reports."""
    for key, X, y in (("validation_rows", Xt_valid, y_valid), ("comparison_rows", Xt_test, y_test)):
        proba = model.predict_proba(X)[:, 1]
        row = evaluate_candidate(name, y, proba)
        row.update(calibration_metrics(y, proba))
        state[key] = [r for r in state.get(key, []) if r["model"] != name] + [row]


def stage_train_final(model_name: str) -> None:
    cfg = load_config()
    state = _load_state()
    params = _load_params()
    Xt_train, y_train = state["Xt_train"], state["y_train"]
    Xt_calib, y_calib = state["Xt_calib"], state["y_calib"]
    Xt_test, y_test = state["Xt_test"], state["y_test"]

    best_params = params.get(model_name, {}).get("best_params", {})
    class_imbalance_strategy = cfg["class_imbalance"]["strategy"]

    feature_names = list(Xt_train.columns)

    def make():
        return build_model(
            model_name,
            best_params,
            y=y_train,
            class_imbalance_strategy=class_imbalance_strategy,
            feature_names=feature_names,
        )

    cv_result = run_stratified_cv(make, Xt_train, y_train, n_folds=cfg["n_cv_folds"], random_state=cfg["random_state"])

    final_model = make()
    final_model.fit(Xt_train, y_train)
    # Prefit calibration: the model keeps its full-training-window fit and
    # only the monotone PD map is learned on the validation window.
    calibrated = calibrate_prefit_model(final_model, Xt_calib, y_calib, method=cfg["calibration"]["method"])

    _record_candidate(state, model_name, calibrated, Xt_calib, y_calib, Xt_test, y_test)
    row = state["comparison_rows"][-1]
    row.update({"cv_roc_auc_mean": cv_result.mean_roc_auc, "cv_roc_auc_std": cv_result.std_roc_auc})

    state["fitted_models"][model_name] = final_model
    state["calibrated_models"][model_name] = calibrated
    state["cv_results"][model_name] = cv_result.summary()
    _save_state(state)

    tracker = _tracker()
    with tracker.start_run(run_name=f"train_final_{model_name}", tags={"stage": "train_final", "model": model_name}):
        tracker.log_params({"model": model_name, **best_params})
        tracker.log_metrics(_numeric_metrics(row))

    logger.info(f"Stage `train_final` complete for {model_name}", extra={"context": row})


def stage_ensemble() -> None:
    cfg = load_config()
    state = _load_state()
    params = _load_params()
    Xt_train, y_train = state["Xt_train"], state["y_train"]
    Xt_calib, y_calib = state["Xt_calib"], state["y_calib"]
    Xt_test, y_test = state["Xt_test"], state["y_test"]
    calibration_method = cfg["calibration"]["method"]
    final_estimator_name = cfg["ensemble"]["stacking_final_estimator"]
    class_imbalance_strategy = cfg["class_imbalance"]["strategy"]

    feature_names = list(Xt_train.columns)

    def make(name: str):
        return build_model(
            name,
            params[name]["best_params"],
            y=y_train,
            class_imbalance_strategy=class_imbalance_strategy,
            feature_names=feature_names,
        )

    base_for_stack = [(name, make(name)) for name in MODEL_NAMES if name in params]

    if len(base_for_stack) < 2:
        logger.warning("Not enough tuned base models for ensembling, skipping.")
        _save_state(state)
        return

    tracker = _tracker()

    if cfg["ensemble"].get("stacking_enabled", True):
        stack = build_stacking_ensemble(base_for_stack, cv=3, final_estimator_name=final_estimator_name)
        stack.fit(Xt_train, y_train)
        # Ensembles are calibrated the same way base models are (stage_train_final)
        # before being eligible for champion selection -- an uncalibrated ensemble's
        # raw predict_proba must never reach production PD/LGD/EAD calculations.
        calibrated_stack = calibrate_prefit_model(stack, Xt_calib, y_calib, method=calibration_method)
        _record_candidate(state, "stacking_ensemble", calibrated_stack, Xt_calib, y_calib, Xt_test, y_test)
        row = state["comparison_rows"][-1]
        state["fitted_models"]["stacking_ensemble"] = stack
        state["calibrated_models"]["stacking_ensemble"] = calibrated_stack
        with tracker.start_run(run_name="ensemble_stacking", tags={"stage": "ensemble", "model": "stacking_ensemble"}):
            tracker.log_params({"base_models": [n for n, _ in base_for_stack], "final_estimator": final_estimator_name})
            tracker.log_metrics(_numeric_metrics(row))
        logger.info("Stacking ensemble evaluated", extra={"context": row})

    if cfg["ensemble"].get("blending_enabled", True):
        base_for_blend = [(n, make(n)) for n in MODEL_NAMES if n in params]
        blend = BlendingEnsemble(
            base_for_blend,
            holdout_fraction=cfg["ensemble"]["blend_holdout_fraction"],
            random_state=cfg["random_state"],
            meta_learner_name=final_estimator_name,
        )
        blend.fit(Xt_train, y_train)
        calibrated_blend = calibrate_prefit_model(blend, Xt_calib, y_calib, method=calibration_method)
        _record_candidate(state, "blending_ensemble", calibrated_blend, Xt_calib, y_calib, Xt_test, y_test)
        row = state["comparison_rows"][-1]
        state["fitted_models"]["blending_ensemble"] = blend
        state["calibrated_models"]["blending_ensemble"] = calibrated_blend
        with tracker.start_run(run_name="ensemble_blending", tags={"stage": "ensemble", "model": "blending_ensemble"}):
            tracker.log_params({"base_models": [n for n, _ in base_for_blend], "meta_learner": final_estimator_name})
            tracker.log_metrics(_numeric_metrics(row))
        logger.info("Blending ensemble evaluated", extra={"context": row})

    _save_state(state)


ENSEMBLE_NAMES = ("stacking_ensemble", "blending_ensemble")


def _select_champion(validation_rows: list[dict], metric: str, min_improvement: float) -> str:
    valid_df = pd.DataFrame(validation_rows).set_index("model")
    singles = valid_df[~valid_df.index.isin(ENSEMBLE_NAMES)]
    best_single = str(singles[metric].idxmax())
    best_overall = str(valid_df[metric].idxmax())
    gain = float(valid_df.loc[best_overall, metric] - valid_df.loc[best_single, metric])
    champion = best_overall if gain >= min_improvement else best_single
    logger.info(
        "Champion selected on validation window",
        extra={
            "context": {
                "champion": champion,
                "best_single": best_single,
                "best_overall": best_overall,
                "metric": metric,
                "ensemble_gain": round(gain, 4),
            }
        },
    )
    return champion


def _write_validation_report(report: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_validation.json").write_text(json.dumps(report, indent=2, default=str))

    m, t, s = report["oot_metrics"], report["threshold_metrics"], report["split"]
    lines = [
        "# Model Validation Report",
        "",
        f"Champion: **{report['model_name']}** (`{report['version']}`)  ",
        f"Generated by `pipelines/run_training.py` at {report['generated_at']}.",
        "",
        "## Data windows (out-of-time split)",
        "",
        "| Window | Dates | Applications | Default rate |",
        "|---|---|---|---|",
    ]
    for name in ("train", "validation", "oot_test"):
        w = s[name]
        lines.append(f"| {name} | {w['start']} to {w['end']} | {w['n']:,} | {w['default_rate']:.2%} |")
    lines += [
        "",
        f"{s['duplicates_removed']} duplicate applications were removed before splitting.",
        "",
        "## Discrimination and calibration (OOT test window)",
        "",
        "| Metric | Value | 95% bootstrap CI |",
        "|---|---|---|",
        f"| ROC-AUC | {m['roc_auc']:.4f} | {m['roc_auc_ci95'][0]:.4f} to {m['roc_auc_ci95'][1]:.4f} |",
        f"| Gini | {m['gini']:.4f} | |",
        f"| KS | {m['ks']:.4f} | {m['ks_ci95'][0]:.4f} to {m['ks_ci95'][1]:.4f} |",
        f"| PR-AUC | {m['pr_auc']:.4f} | |",
        f"| Brier score | {m['brier_score']:.4f} | |",
        f"| Expected calibration error | {m['expected_calibration_error']:.4f} | |",
        f"| Mean predicted PD vs observed | {m['mean_predicted_pd']:.2%} vs {m['observed_default_rate']:.2%} | |",
        f"| Score PSI (train to OOT) | {report['score_psi_train_vs_oot']:.4f} | |",
        "",
        "## Decision threshold",
        "",
        f"Cost-sensitive threshold chosen on the validation window: **PD >= {t['threshold']:.4f} is declined** "
        f"(a missed default costs {report['cost_matrix']['false_negative']:g}x a wrongly declined good borrower).",
        "",
        f"On the OOT window: approval rate {t['approval_rate']:.1%}, defaults caught (recall) {t['recall']:.1%}, "
        f"precision of declines {t['precision']:.1%}.",
        "",
        "## Gains table (OOT, riskiest decile first)",
        "",
        "| Decile | Applications | Defaults | Bad rate | Cumulative capture | Lift |",
        "|---|---|---|---|---|---|",
    ]
    for d in report["decile_table"]:
        lines.append(
            f"| {d['decile']} | {d['n']:,} | {d['defaults']} | {d['bad_rate']:.2%} | "
            f"{d['capture_rate']:.1%} | {d['lift']:.2f} |"
        )
    lines += ["", "## Fair-lending review (OOT window, four-fifths rule)", ""]
    for attr, a in report["fairness"]["attributes"].items():
        lines += [
            f"**{attr}**: minimum adverse impact ratio {a['min_adverse_impact_ratio']:.3f} "
            f"({'pass' if a['passes'] else 'needs review'})",
            "",
            "| Group | n | Approval rate | AIR | Observed DR | Mean PD | AUC |",
            "|---|---|---|---|---|---|---|",
        ]
        for g in a["groups"]:
            auc = f"{g['roc_auc']:.3f}" if g["roc_auc"] is not None else "n/a"
            lines.append(
                f"| {g['group']} | {g['n']:,} | {g['approval_rate']:.1%} | {g['adverse_impact_ratio']:.3f} | "
                f"{g['observed_default_rate']:.2%} | {g['mean_predicted_pd']:.2%} | {auc} |"
            )
        lines.append("")
    lines += [
        "## Candidate models (OOT window; selection used the validation window)",
        "",
        "| Model | ROC-AUC | PR-AUC | Brier | ECE |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(report["candidates"], key=lambda r: -r["roc_auc"]):
        lines.append(
            f"| {r['model']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['brier_score']:.4f} | "
            f"{r['expected_calibration_error']:.4f} |"
        )
    (REPORTS_DIR / "MODEL_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_finalize() -> None:
    cfg = load_config()
    settings = get_settings()
    state = _load_state()
    Xt_calib, y_calib = state["Xt_calib"], state["y_calib"]
    Xt_test, y_test = state["Xt_test"], state["y_test"]

    comparison_df = build_comparison_table(state["comparison_rows"])
    comparison_path = CHECKPOINT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    best_name = _select_champion(
        state["validation_rows"],
        cfg["selection_metric"],
        cfg["ensemble"].get("min_improvement", 0.0),
    )
    best_model = state["calibrated_models"].get(best_name) or state["fitted_models"][best_name]

    # Threshold is chosen on the validation window and only *evaluated* on OOT.
    threshold_result = optimize_threshold(
        y_calib,
        best_model.predict_proba(Xt_calib)[:, 1],
        method=cfg["threshold_optimization"]["method"],
        cost_fn=cfg["cost_sensitive"]["cost_false_negative"],
        cost_fp=cfg["cost_sensitive"]["cost_false_positive"],
    )
    test_proba = best_model.predict_proba(Xt_test)[:, 1]
    train_proba = best_model.predict_proba(state["Xt_train"])[:, 1]
    threshold_metrics = evaluate_at_threshold(y_test, test_proba, threshold_result.threshold)

    test_row = next(r for r in state["comparison_rows"] if r["model"] == best_name)
    oot_metrics = {**test_row, **credit_validation_metrics(y_test, test_proba)}
    fairness = disparate_impact_report(state["X_test_raw"], y_test, test_proba, threshold_result.threshold)
    score_psi = compute_numeric_psi(pd.Series(train_proba), pd.Series(test_proba))

    tracker = _tracker()
    with tracker.start_run(run_name=f"champion_{best_name}", tags={"stage": "finalize"}):
        tracker.log_params({"champion_model": best_name, "threshold": threshold_result.threshold})
        tracker.log_metrics(_numeric_metrics({**threshold_metrics, **oot_metrics, "score_psi": score_psi}))
        tracker.log_artifact(str(comparison_path))

    registry = ModelRegistry(settings.model_artifact_dir)
    version = f"v_{best_name}_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%S')}"
    registry.register_version(
        version=version,
        model=best_model,
        feature_pipeline=state["feature_pipeline"],
        metadata={
            "model_name": best_name,
            "decision_threshold": threshold_result.threshold,
            "test_metrics": oot_metrics,
            "threshold_metrics": threshold_metrics,
            "split": state["split"],
            "fairness_passes": fairness["passes"],
        },
    )
    state["background_raw"].to_csv(
        settings.model_artifact_dir / version / explainer_cache.BACKGROUND_FILENAME, index=False
    )
    registry.set_champion(version)

    report = {
        "model_name": best_name,
        "version": version,
        "generated_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "split": state["split"],
        "oot_metrics": oot_metrics,
        "threshold_metrics": threshold_metrics,
        "cost_matrix": {
            "false_negative": cfg["cost_sensitive"]["cost_false_negative"],
            "false_positive": cfg["cost_sensitive"]["cost_false_positive"],
        },
        "score_psi_train_vs_oot": round(float(score_psi), 4),
        "decile_table": decile_table(y_test, test_proba).to_dict(orient="records"),
        "fairness": fairness,
        "candidates": state["comparison_rows"],
        "validation_candidates": state["validation_rows"],
    }
    _write_validation_report(report)

    logger.info(
        "Stage `finalize` complete",
        extra={"context": {"champion": best_name, "version": version, "threshold": threshold_result.threshold}},
    )
    print("\n=== MODEL COMPARISON (OOT window) ===")
    print(comparison_df.to_string(index=False))
    print(f"\nChampion: {best_name} (version {version})")
    print(f"OOT: AUC {oot_metrics['roc_auc']}  Gini {oot_metrics['gini']}  KS {oot_metrics['ks']}")
    print(f"Threshold {threshold_result.threshold:.4f}: {threshold_metrics}")
    print(f"Fair-lending four-fifths check: {'PASS' if fairness['passes'] else 'REVIEW'}")
    print(f"Validation report: {REPORTS_DIR / 'MODEL_VALIDATION.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit risk training pipeline orchestrator.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["prepare", "tune", "train_final", "ensemble", "finalize", "all"],
    )
    parser.add_argument("--model", type=str, default=None, help="Model name for `tune`/`train_final` stages.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "credit_applications.csv"),
    )
    args = parser.parse_args()

    if args.stage in ("prepare", "all") and not Path(args.data_path).exists():
        # Fresh clone / fresh container: the dataset is gitignored and
        # docker-ignored, so build the synthetic reference set on first run.
        from data.generate_synthetic_data import generate_credit_dataset

        logger.info(
            "No training data found, generating the synthetic dataset", extra={"context": {"path": args.data_path}}
        )
        Path(args.data_path).parent.mkdir(parents=True, exist_ok=True)
        generate_credit_dataset().to_csv(args.data_path, index=False)

    if args.stage == "prepare":
        stage_prepare(args.data_path)
    elif args.stage == "tune":
        stage_tune(args.model)
    elif args.stage == "train_final":
        stage_train_final(args.model)
    elif args.stage == "ensemble":
        stage_ensemble()
    elif args.stage == "finalize":
        stage_finalize()
    elif args.stage == "all":
        stage_prepare(args.data_path)
        for m in MODEL_NAMES:
            stage_tune(m)
            stage_train_final(m)
        stage_ensemble()
        stage_finalize()


if __name__ == "__main__":
    main()
