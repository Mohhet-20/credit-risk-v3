# Model Documentation

This is the model's methodology document, organised the way a model-risk
reviewer (SR 11-7 / OCC 2011-12) reads one: purpose, data, design,
validation, fair lending, explainability and limitations. The numbers for the
current champion are in [`reports/MODEL_VALIDATION.md`](../reports/MODEL_VALIDATION.md),
which the training pipeline regenerates on every run. The API serves the same
numbers at `GET /api/v1/models/validation-report`, and the web console shows
them on its **Model Validation** page.

## Purpose and use

The model estimates the **probability of default (PD)** of an unsecured
consumer instalment loan application at origination. The PD drives three
things:

1. **The decision.** Applications with PD at or above a cost-optimised cut-off
   are declined, and the rest are approved.
2. **The risk band and scorecard points** shown to loan officers.
3. **Adverse-action reasons**, which are required for every decline.

It is not a pricing, limit-setting or behavioural model.

## Data

Real consumer credit data is regulated (FCRA, GLBA, GDPR) and cannot be
published, so the platform trains on a **synthetic, causally structured**
dataset (`data/generate_synthetic_data.py`, 50,000 applications by default):

- Default is drawn from an explicit logistic function of credit score, DTI,
  utilisation, delinquencies, derogatory marks, inquiries, loan-to-income,
  interest rate, employment and housing, plus label noise. The relationships
  are therefore realistic and known, so SHAP, PDPs and calibration can be
  checked against ground truth.
- **Age, marital status and state have no causal effect.** Any dependence the
  model picked up on them would be spurious, which is what the fair-lending
  review tests for.
- **Macro drift:** later vintages default slightly more, which makes the
  out-of-time test a genuine test of stability.
- Injected MAR missingness, heavy-tailed outliers and duplicate applications
  exercise the data-quality layer.

## Validation design

| Window | Share | Used for |
|---|---|---|
| Train | oldest 60% of applications by `application_date` | feature-pipeline fit, Optuna tuning, final fit |
| Validation | next 20% | probability calibration, champion selection, decision threshold |
| Out-of-time (OOT) test | newest 20% | reported once; never used for any choice |

- **The split is out-of-time, not random.** A random split lets a model learn
  from loans originated after the ones it is tested on and overstates
  performance.
- **Duplicates are removed before splitting** (by `application_id`), so no
  loan sits on both sides of the boundary.
- **Calibration is prefit.** Each model keeps its fit on the full training
  window, and only the monotone PD map (isotonic by default) is learned on the
  validation window. (An earlier version used `CalibratedClassifierCV(cv=5)`,
  which refits clones on the calibration split alone and quietly discards the
  training-window fit.)
- **OOT metrics** (`training/credit_metrics.py`): ROC-AUC with a 95% bootstrap
  CI, Gini, KS with a 95% bootstrap CI, PR-AUC, Brier score, expected
  calibration error, mean predicted PD against the observed default rate, a
  decile gains/lift table, and PSI of the score distribution between train
  and OOT.

## Model zoo and selection

Five families are tuned with Optuna (TPE sampler with a median pruner, 3-fold
stratified CV on ROC-AUC, budget in `configs/model_training.yaml`): logistic
regression, random forest, XGBoost, LightGBM and CatBoost. Stacking and
blending ensembles are built over the tuned models.

- **Selection metric:** PR-AUC on the validation window. Under class
  imbalance it is more informative than ROC-AUC.
- **Parsimony rule:** an ensemble becomes champion only if it beats the best
  single model by at least `ensemble.min_improvement` (0.005). Otherwise the
  single model wins. It is simpler to validate and monitor, and TreeSHAP
  explains it exactly.
- **Monotonic constraints** on XGBoost, LightGBM and CatBoost
  (`training/tuning.py::MONOTONE_DIRECTIONS`) guarantee business-sensible
  behaviour. For example, PD can never rise as the credit score rises, or fall
  as utilisation, DTI, delinquencies or inquiries rise. Tests check this on a
  trained model and end to end through the feature pipeline, not only on
  the configuration.
- **Class imbalance** is handled with class weights (`class_imbalance.strategy:
  class_weight`). SMOTE is deliberately not implemented, because synthetic
  minority rows distort calibrated PDs. Selecting it raises
  `NotImplementedError`.

## Decision threshold

`training/threshold_optimization.py` chooses the cut-off **on the validation
window** to minimise expected cost, using the cost matrix
`cost_false_negative: 5.0` and `cost_false_positive: 1.0`: approving a
borrower who defaults costs five times as much as declining one who would
have repaid. `f1` and `youden_j` are available as alternatives. The OOT window
then reports the approval rate, the share of defaults caught and the precision
of declines at that cut-off.

## Fair lending

- **Excluded from the model:** `age` and `marital_status` (protected bases
  under ECOA) and `state` (geography is the textbook redlining proxy). See
  `core/constants.py::MODEL_EXCLUDED_FEATURES`. `FeaturePipeline` drops them
  before any fitting, so not even an imputation statistic is learned from
  them. Tests check that changing them leaves the prediction byte-for-byte
  unchanged.
- **Retained for monitoring.** They stay on the application record so that
  `analytics/fairness.py` can test decisions for **disparate impact**. For
  each age band (18-39, 40-61, 62+, where 62 is Reg B's "elderly" threshold)
  and each marital status, the analysis reports the approval rate, the adverse
  impact ratio (AIR) against the most-favoured group, the four-fifths-rule
  flag (AIR below 0.80), observed default rate against mean predicted PD, and
  within-group AUC.
- An AIR flag is a trigger for review, not a verdict. A gap that tracks a gap
  in observed default rates and is calibrated within each group points to
  legitimate risk differences carried by permitted features. The next step in
  a real deployment would be a less-discriminatory-alternative search.

## Explainability and adverse action

- **SHAP** (`explainability/explanations.py`): TreeSHAP for tree models,
  including inside the calibration wrapper (calibration is monotone, so
  attribution rankings carry over). A permutation explainer is the fallback
  for ensembles. Explainers are cached per model version
  (`explainability/explainer_cache.py`). The SHAP background is a sample of
  the model's own training window, saved next to the artifact.
- **Adverse-action reasons** (`explainability/adverse_action.py`) cover ECOA
  and Reg B (12 CFR 1002.9). Every model feature maps to one of 14 reason
  groups, for example *"Proportion of revolving balances to credit limits is
  too high"*. SHAP values are summed within each group, and the top four
  groups that **raised** risk are returned with a stable code. A factor that
  helped the applicant can never be cited. Reasons are returned on every
  declined `/predict` response and written to the audit log.
- **Scorecard points** (`core/scorecard.py`) use points-to-double-odds
  scaling: 600 points means 50:1 good:bad odds, and every 20 points doubles
  the odds, clipped to 300-850. The mapping is a monotone presentation layer
  and does not affect decisions.
- Also available: LIME, counterfactuals ("smallest change that flips the
  decision"), plain-English narratives and per-applicant PDF reports.

## Risk bands

| Band | PD range |
|---|---|
| Very Low | 0.00 to 0.05 |
| Low | 0.05 to 0.15 |
| Medium | 0.15 to 0.35 |
| High | 0.35 to 0.60 |
| Very High | 0.60 to 1.00 |

## Feature pipeline

`feature_store/pipeline.py::FeaturePipeline` runs these steps in order:

1. Drop the protected and proxy attributes.
2. Impute missing values and add missingness indicators.
3. Winsorise outliers using IQR bounds.
4. Add domain features (loan-to-income, payment-to-income, delinquency
   severity, liquidity ratios and an utilisation × low-score interaction).
   Score and utilisation *bands* are computed for reporting but are not model
   inputs. As unconstrained step-function copies of constrained features,
   they would let the model route around the monotonic constraints. An
   end-to-end smoke test caught exactly that: PD rose as the score went from
   750 to 800. A test now sweeps the raw score through the full pipeline.
5. Apply K-fold out-of-fold target encoding to `loan_purpose`.
6. One-hot encode the remaining categoricals.
7. Select features by variance and correlation.

The fitted pipeline is versioned with the model, so training and serving
transformations cannot diverge.

## Monitoring

`/monitoring/*` reports prediction volume, the PD and risk-band
distributions, latency percentiles, and PSI drift per input feature against
the training reference. The validation report records the train-to-OOT score
PSI as a baseline. Rule of thumb: PSI below 0.10 is stable, 0.10 to 0.25 is
worth watching, and above 0.25 calls for investigation or retraining.

## Limitations

- **Synthetic data.** The figures show the methodology working on known
  ground truth; they are not evidence of real-world performance. A real
  deployment would revalidate on bureau and performance data, add
  reject inference, and set a reporting horizon such as 12-month default.
- **The cost matrix is illustrative.** A real cut-off would come from LGD,
  margin and funding cost, and would be signed off by credit policy.
- **No champion/challenger traffic split.** Promotion swaps the champion
  atomically, and rollback is a pointer rewrite.

## Experiment tracking and registry

Every run logs parameters, metrics and artifacts to MLflow under
`credit-risk-v2`. Promotion copies the model, the feature pipeline, the SHAP
background sample and the metadata (OOT metrics, threshold, split windows and
fair-lending result) into `models/artifacts/<version>/`. The API serves from
that copy, so serving never depends on MLflow being up. See
[`ARCHITECTURE.md`](ARCHITECTURE.md).
