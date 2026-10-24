"use client";
import * as React from "react";
import { CheckCircle2, Loader2, ShieldAlert, XCircle } from "lucide-react";
import { PageHeader } from "@/components/dashboard/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useScoreApplication } from "@/hooks/use-credit";
import { RISK_BAND_LABELS, type CreditApplication, type PredictionResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

type NumericField = {
  [K in keyof CreditApplication]-?: CreditApplication[K] extends number ? K : never;
}[keyof CreditApplication];

const DEFAULT_APPLICATION: CreditApplication = {
  customer_id: "",
  age: 38,
  annual_income: 62000,
  employment_length_years: 4,
  loan_amount: 15000,
  loan_term_months: 36,
  interest_rate: 11.5,
  credit_score: 690,
  debt_to_income_ratio: 18,
  num_open_accounts: 6,
  num_credit_inquiries_6m: 1,
  revolving_utilization: 0.35,
  num_delinquencies_2yrs: 0,
  months_since_last_delinquency: 999,
  num_derogatory_marks: 0,
  total_credit_limit: 18000,
  savings_balance: 4500,
  checking_balance: 1800,
  num_dependents: 1,
  years_at_residence: 3,
  home_ownership: "RENT",
  loan_purpose: "debt_consolidation",
  employment_status: "employed",
  education_level: "bachelors",
  marital_status: "single",
  state: "CA",
};

const SECTIONS: { title: string; fields: { key: NumericField; label: string; step?: string; hint?: string }[] }[] = [
  {
    title: "Loan request",
    fields: [
      { key: "loan_amount", label: "Loan amount ($)" },
      { key: "loan_term_months", label: "Term (months)" },
      { key: "interest_rate", label: "Interest rate (%)", step: "0.01" },
    ],
  },
  {
    title: "Income & employment",
    fields: [
      { key: "annual_income", label: "Annual income ($)" },
      { key: "employment_length_years", label: "Years employed", step: "0.1" },
      { key: "debt_to_income_ratio", label: "Debt-to-income (%)", step: "0.1" },
      { key: "savings_balance", label: "Savings ($)" },
      { key: "checking_balance", label: "Checking ($)" },
    ],
  },
  {
    title: "Credit bureau",
    fields: [
      { key: "credit_score", label: "Bureau score" },
      { key: "revolving_utilization", label: "Revolving utilization", step: "0.01", hint: "0 to 1" },
      { key: "total_credit_limit", label: "Total credit limit ($)" },
      { key: "num_open_accounts", label: "Open accounts" },
      { key: "num_credit_inquiries_6m", label: "Inquiries (6m)" },
      { key: "num_delinquencies_2yrs", label: "Delinquencies (2y)" },
      { key: "months_since_last_delinquency", label: "Months since delinquency", hint: "999 = never" },
      { key: "num_derogatory_marks", label: "Derogatory marks" },
    ],
  },
  {
    title: "Applicant",
    fields: [
      { key: "age", label: "Age", hint: "Fair-lending monitoring only" },
      { key: "num_dependents", label: "Dependents" },
      { key: "years_at_residence", label: "Years at residence", step: "0.1" },
    ],
  },
];

const SELECTS: { key: keyof CreditApplication; label: string; options: string[]; hint?: string }[] = [
  {
    key: "loan_purpose",
    label: "Loan purpose",
    options: ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "medical", "small_business", "car", "other"],
  },
  { key: "home_ownership", label: "Home ownership", options: ["OWN", "MORTGAGE", "RENT", "OTHER"] },
  { key: "employment_status", label: "Employment status", options: ["employed", "self_employed", "unemployed", "retired"] },
  { key: "education_level", label: "Education", options: ["high_school", "bachelors", "masters", "doctorate", "other"] },
  { key: "marital_status", label: "Marital status", options: ["single", "married", "divorced", "widowed"], hint: "Fair-lending monitoring only" },
  { key: "state", label: "State", options: ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"], hint: "Fair-lending monitoring only" },
];

const selectClass =
  "flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]";

const pretty = (s: string) => s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

export default function ScoreApplicationPage() {
  const [form, setForm] = React.useState<CreditApplication>(DEFAULT_APPLICATION);
  const score = useScoreApplication();

  const setNumber = (key: NumericField, value: string) =>
    setForm((f) => ({ ...f, [key]: value === "" ? ("" as unknown as number) : Number(value) }));

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    score.mutate({ ...form, customer_id: form.customer_id || null });
  };

  return (
    <div>
      <PageHeader
        title="Score an application"
        description="Calibrated probability of default, scorecard points, the cost-optimised decision and, for declines, the adverse-action reasons."
      />

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        <form onSubmit={onSubmit} className="xl:col-span-3">
          <Card>
            <CardContent className="space-y-6 py-5">
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <Label htmlFor="customer_id">Customer ID (optional)</Label>
                  <Input
                    id="customer_id"
                    value={form.customer_id ?? ""}
                    onChange={(e) => setForm((f) => ({ ...f, customer_id: e.target.value }))}
                    placeholder="CUST-0001234"
                  />
                </div>
              </div>

              {SECTIONS.map((section) => (
                <fieldset key={section.title} className="space-y-3">
                  <legend className="text-sm font-semibold">{section.title}</legend>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                    {section.fields.map((f) => (
                      <div key={f.key} className="space-y-1.5">
                        <Label htmlFor={f.key}>{f.label}</Label>
                        <Input
                          id={f.key}
                          type="number"
                          inputMode="decimal"
                          step={f.step ?? "1"}
                          required
                          value={form[f.key] as number}
                          onChange={(e) => setNumber(f.key, e.target.value)}
                        />
                        {f.hint && <p className="text-xs text-muted-foreground">{f.hint}</p>}
                      </div>
                    ))}
                  </div>
                </fieldset>
              ))}

              <fieldset className="space-y-3">
                <legend className="text-sm font-semibold">Profile</legend>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {SELECTS.map((s) => (
                    <div key={s.key} className="space-y-1.5">
                      <Label htmlFor={s.key}>{s.label}</Label>
                      <select
                        id={s.key}
                        className={selectClass}
                        value={form[s.key] as string}
                        onChange={(e) => setForm((f) => ({ ...f, [s.key]: e.target.value }))}
                      >
                        {s.options.map((o) => (
                          <option key={o} value={o}>
                            {s.key === "state" || s.key === "home_ownership" ? o : pretty(o)}
                          </option>
                        ))}
                      </select>
                      {s.hint && <p className="text-xs text-muted-foreground">{s.hint}</p>}
                    </div>
                  ))}
                </div>
              </fieldset>

              <div className="flex items-center gap-3">
                <Button type="submit" disabled={score.isPending}>
                  {score.isPending && <Loader2 className="size-4 animate-spin" />}
                  Score application
                </Button>
                <Button type="button" variant="ghost" onClick={() => setForm(DEFAULT_APPLICATION)}>
                  Reset
                </Button>
              </div>
            </CardContent>
          </Card>
        </form>

        <div className="xl:col-span-2">
          {score.isError && (
            <Card className="border-danger/40">
              <CardContent className="py-5 text-sm text-danger">{score.error.message}</CardContent>
            </Card>
          )}
          {score.data ? <DecisionCard result={score.data} /> : !score.isError && <EmptyDecision />}
        </div>
      </div>
    </div>
  );
}

function EmptyDecision() {
  return (
    <Card className="h-full">
      <CardContent className="flex h-full min-h-48 flex-col items-center justify-center gap-2 py-10 text-center text-sm text-muted-foreground">
        <ShieldAlert className="size-6" />
        Submit an application to see the decision.
      </CardContent>
    </Card>
  );
}

function DecisionCard({ result }: { result: PredictionResponse }) {
  const approved = result.decision === "approved";
  const pd = result.predicted_probability;
  const markerLeft = `${Math.min(pd / Math.max(result.decision_threshold * 2, 0.0001), 1) * 100}%`;

  return (
    <Card className="sticky top-4">
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-lg">
            {approved ? <CheckCircle2 className="size-5 text-success" /> : <XCircle className="size-5 text-danger" />}
            {approved ? "Approved" : "Declined"}
          </CardTitle>
          <Badge variant={approved ? "success" : "danger"}>{RISK_BAND_LABELS[result.risk_band] ?? result.risk_band} risk</Badge>
        </div>
        <CardDescription>
          {result.model_name} · {result.model_version} · {result.latency_ms.toFixed(0)} ms
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5 pb-5">
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Probability of default" value={`${(pd * 100).toFixed(2)}%`} />
          <Stat label="Credit score" value={String(result.credit_score)} hint="600 = 50:1 odds, +20 doubles" />
        </div>

        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>0%</span>
            <span>Decline at PD ≥ {(result.decision_threshold * 100).toFixed(1)}%</span>
          </div>
          <div className="relative h-2 rounded-full bg-gradient-to-r from-success/60 via-warning/60 to-danger/70">
            <div className="absolute left-1/2 top-[-4px] h-4 w-px bg-foreground/60" aria-hidden />
            <div
              className="absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-background bg-foreground shadow"
              style={{ left: markerLeft }}
              aria-label={`PD ${(pd * 100).toFixed(2)}%`}
            />
          </div>
        </div>

        {!approved && (
          <div className="space-y-2">
            <h3 className="text-sm font-semibold">Principal reasons for decline</h3>
            <p className="text-xs text-muted-foreground">ECOA / Regulation B adverse-action reasons, strongest first.</p>
            {result.adverse_action_reasons.length === 0 ? (
              <p className="text-sm text-warning">Reason generation failed; this decline needs manual reasons.</p>
            ) : (
              <ol className="space-y-2">
                {result.adverse_action_reasons.map((r) => (
                  <li key={r.code} className="flex gap-3 rounded-lg border p-3">
                    <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-danger/10 text-xs font-semibold text-danger">
                      {r.rank}
                    </span>
                    <div className="min-w-0">
                      <p className="text-sm font-medium">{r.reason}</p>
                      <p className="truncate text-xs text-muted-foreground">
                        {r.code} · {r.features.join(", ")}
                      </p>
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className={cn("rounded-lg border p-3")}>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-2xl font-semibold tabular-nums tracking-tight">{value}</p>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}
