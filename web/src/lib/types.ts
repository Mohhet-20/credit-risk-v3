// Mirrors the Pydantic schemas in api/schemas.py -- kept in one file so a
// backend contract change is a one-file diff on the frontend too.

export type UserRole =
  | "super_admin"
  | "admin"
  | "ml_engineer"
  | "risk_analyst"
  | "auditor"
  | "loan_officer"
  | "viewer";

export interface UserOut {
  id: string;
  username: string;
  email: string;
  role: UserRole;
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface LoginResponse {
  access_token: string | null;
  refresh_token: string | null;
  mfa_required: boolean;
  mfa_challenge_token?: string | null;
}

export interface HealthResponse {
  status: string;
  champion_model: string | null;
  champion_version: string | null;
}

export interface MonitoringStats {
  n_predictions: number;
  avg_predicted_probability: number | null;
  avg_latency_ms: number | null;
  window_days: number;
}

export interface PredictionDistributionBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface PredictionDistribution {
  n: number;
  histogram: PredictionDistributionBin[];
  risk_band_counts: Record<string, number>;
  decision_counts: Record<string, number>;
}

export interface LatencyStats {
  n: number;
  p50_ms?: number;
  p95_ms?: number;
  p99_ms?: number;
  max_ms?: number;
}

export interface ModelTestMetrics {
  rank?: number;
  model?: string;
  roc_auc?: number;
  pr_auc?: number;
  brier_score?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  expected_calibration_error?: number;
  cv_roc_auc_mean?: number;
  cv_roc_auc_std?: number;
  gini?: number;
  ks?: number;
}

export interface ModelVersionOut {
  version: string;
  model_name: string;
  registered_at: string;
  decision_threshold: number | null;
  test_metrics: ModelTestMetrics | null;
  is_champion: boolean;
}

export interface PredictionHistoryItem {
  prediction_id: string;
  customer_id: string | null;
  predicted_probability: number;
  risk_band: string;
  decision: string;
  model_name: string;
  model_version: string;
  created_at: string | null;
}

export const RISK_BAND_LABELS: Record<string, string> = {
  very_low: "Very Low",
  low: "Low",
  medium: "Medium",
  high: "High",
  very_high: "Very High",
};

export interface CreditApplication {
  customer_id?: string | null;
  age: number;
  annual_income: number;
  employment_length_years: number;
  loan_amount: number;
  loan_term_months: number;
  interest_rate: number;
  credit_score: number;
  debt_to_income_ratio: number;
  num_open_accounts: number;
  num_credit_inquiries_6m: number;
  revolving_utilization: number;
  num_delinquencies_2yrs: number;
  months_since_last_delinquency: number;
  num_derogatory_marks: number;
  total_credit_limit: number;
  savings_balance: number;
  checking_balance: number;
  num_dependents: number;
  years_at_residence: number;
  home_ownership: "OWN" | "MORTGAGE" | "RENT" | "OTHER";
  loan_purpose: string;
  employment_status: "employed" | "self_employed" | "unemployed" | "retired";
  education_level: "high_school" | "bachelors" | "masters" | "doctorate" | "other";
  marital_status: "single" | "married" | "divorced" | "widowed";
  state: string;
}

export interface AdverseActionReason {
  rank: number;
  code: string;
  reason: string;
  contribution: number;
  features: string[];
}

export interface PredictionResponse {
  prediction_id: string;
  customer_id: string | null;
  predicted_probability: number;
  credit_score: number;
  risk_band: string;
  decision: "approved" | "rejected" | string;
  decision_threshold: number;
  adverse_action_reasons: AdverseActionReason[];
  model_name: string;
  model_version: string;
  latency_ms: number;
}

export interface DataWindow {
  start: string;
  end: string;
  n: number;
  default_rate: number;
}

export interface FairnessGroup {
  group: string;
  n: number;
  approval_rate: number;
  observed_default_rate: number;
  mean_predicted_pd: number;
  roc_auc: number | null;
  adverse_impact_ratio: number;
  passes_four_fifths: boolean;
}

export interface ValidationReport {
  model_name: string;
  version: string;
  generated_at: string;
  split: { train: DataWindow; validation: DataWindow; oot_test: DataWindow; duplicates_removed: number };
  oot_metrics: ModelTestMetrics & {
    gini: number;
    ks: number;
    roc_auc_ci95: [number, number];
    ks_ci95: [number, number];
    observed_default_rate: number;
    mean_predicted_pd: number;
    n: number;
    n_defaults: number;
  };
  threshold_metrics: {
    threshold: number;
    precision: number;
    recall: number;
    approval_rate: number;
    true_positives: number;
    false_positives: number;
    true_negatives: number;
    false_negatives: number;
  };
  cost_matrix: { false_negative: number; false_positive: number };
  score_psi_train_vs_oot: number;
  decile_table: {
    decile: number;
    n: number;
    defaults: number;
    bad_rate: number;
    capture_rate: number;
    lift: number;
    min_pd: number;
    max_pd: number;
  }[];
  fairness: {
    passes: boolean;
    four_fifths_threshold: number;
    attributes: Record<string, { groups: FairnessGroup[]; min_adverse_impact_ratio: number; passes: boolean }>;
  };
  candidates: (ModelTestMetrics & { model: string })[];
}
