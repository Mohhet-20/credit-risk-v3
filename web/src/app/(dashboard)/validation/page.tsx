"use client";
import { CheckCircle2, AlertTriangle } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Line, ComposedChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { PageHeader } from "@/components/dashboard/page-header";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useValidationReport } from "@/hooks/use-credit";
import type { ValidationReport } from "@/lib/types";

const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;

export default function ModelValidationPage() {
  const { data, isLoading, error } = useValidationReport();

  return (
    <div>
      <PageHeader
        title="Model validation"
        description="Out-of-time evidence for the champion model: discrimination, calibration, decision threshold and fair-lending review."
      />
      {isLoading && <Skeleton className="h-96 w-full" />}
      {error && (
        <Card>
          <CardContent className="py-6 text-sm text-muted-foreground">{error.message}</CardContent>
        </Card>
      )}
      {data && <Report report={data} />}
    </div>
  );
}

function Report({ report }: { report: ValidationReport }) {
  const m = report.oot_metrics;
  const t = report.threshold_metrics;
  const s = report.split;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Metric label="ROC-AUC (OOT)" value={m.roc_auc!.toFixed(3)} hint={`95% CI ${m.roc_auc_ci95[0].toFixed(3)} to ${m.roc_auc_ci95[1].toFixed(3)}`} />
        <Metric label="Gini" value={m.gini.toFixed(3)} hint="2 × AUC − 1" />
        <Metric label="KS" value={m.ks.toFixed(3)} hint={`95% CI ${m.ks_ci95[0].toFixed(3)} to ${m.ks_ci95[1].toFixed(3)}`} />
        <Metric
          label="Calibration"
          value={pct(m.mean_predicted_pd, 2)}
          hint={`predicted vs ${pct(m.observed_default_rate, 2)} observed`}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Gains by score decile</CardTitle>
            <CardDescription>OOT window, riskiest decile first. Bars: bad rate. Line: cumulative share of defaults captured.</CardDescription>
          </CardHeader>
          <CardContent className="h-72 pb-5">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={report.decile_table} margin={{ left: -10, right: 8 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border" />
                <XAxis dataKey="decile" tickLine={false} axisLine={false} fontSize={12} />
                <YAxis yAxisId="l" tickFormatter={(v) => pct(v, 0)} tickLine={false} axisLine={false} fontSize={12} />
                <YAxis yAxisId="r" orientation="right" domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} tickLine={false} axisLine={false} fontSize={12} />
                <Tooltip formatter={(v) => pct(Number(v))} labelFormatter={(d) => `Decile ${d}`} />
                <Bar yAxisId="l" dataKey="bad_rate" name="Bad rate" fill="var(--color-danger)" radius={[4, 4, 0, 0]} />
                <Line yAxisId="r" dataKey="capture_rate" name="Cumulative capture" stroke="var(--color-primary)" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Decision threshold</CardTitle>
            <CardDescription>
              Chosen on the validation window to minimise cost (a missed default costs {report.cost_matrix.false_negative}× a wrong decline).
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 pb-5 text-sm">
            <Row k="Decline when PD ≥" v={pct(t.threshold, 2)} />
            <Row k="Approval rate (OOT)" v={pct(t.approval_rate)} />
            <Row k="Defaults caught" v={pct(t.recall)} />
            <Row k="Precision of declines" v={pct(t.precision)} />
            <Row k="Score PSI, train to OOT" v={report.score_psi_train_vs_oot.toFixed(3)} />
            <Row k="Brier score" v={m.brier_score!.toFixed(4)} />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <CardTitle>Fair-lending review</CardTitle>
            <Badge variant={report.fairness.passes ? "success" : "warning"}>
              {report.fairness.passes ? "Passes four-fifths rule" : "Needs review"}
            </Badge>
          </div>
          <CardDescription>
            Age, marital status and state are never model inputs. They are used only here, to test decisions for disparate impact
            (adverse impact ratio below {report.fairness.four_fifths_threshold} is flagged).
          </CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-6 pb-5 lg:grid-cols-2">
          {Object.entries(report.fairness.attributes).map(([attr, a]) => (
            <div key={attr} className="space-y-2">
              <div className="flex items-center gap-2 text-sm font-medium">
                {a.passes ? <CheckCircle2 className="size-4 text-success" /> : <AlertTriangle className="size-4 text-warning" />}
                {attr.replace(/_/g, " ")}
              </div>
              <div className="h-40">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={a.groups} margin={{ left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border" />
                    <XAxis dataKey="group" tickLine={false} axisLine={false} fontSize={12} />
                    <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} tickLine={false} axisLine={false} fontSize={12} />
                    <Tooltip formatter={(v) => Number(v).toFixed(3)} />
                    <Bar dataKey="adverse_impact_ratio" name="Adverse impact ratio" fill="var(--color-primary)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <table className="w-full text-xs">
                <thead className="text-muted-foreground">
                  <tr className="text-left">
                    <th className="py-1 font-medium">Group</th>
                    <th className="font-medium">n</th>
                    <th className="font-medium">Approved</th>
                    <th className="font-medium">Observed DR</th>
                    <th className="font-medium">Mean PD</th>
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {a.groups.map((g) => (
                    <tr key={g.group} className="border-t">
                      <td className="py-1">{g.group}</td>
                      <td>{g.n.toLocaleString()}</td>
                      <td>{pct(g.approval_rate)}</td>
                      <td>{pct(g.observed_default_rate, 2)}</td>
                      <td>{pct(g.mean_predicted_pd, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Out-of-time data windows</CardTitle>
            <CardDescription>{s.duplicates_removed} duplicate applications removed before splitting.</CardDescription>
          </CardHeader>
          <CardContent className="pb-5">
            <table className="w-full text-sm">
              <tbody className="tabular-nums">
                {(["train", "validation", "oot_test"] as const).map((k) => (
                  <tr key={k} className="border-t first:border-t-0">
                    <td className="py-1.5 font-medium">{k === "oot_test" ? "OOT test" : k[0].toUpperCase() + k.slice(1)}</td>
                    <td className="text-muted-foreground">
                      {s[k].start} to {s[k].end}
                    </td>
                    <td className="text-right">{s[k].n.toLocaleString()}</td>
                    <td className="text-right">{pct(s[k].default_rate, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Candidate models (OOT)</CardTitle>
            <CardDescription>Champion selected on the validation window: {report.model_name}</CardDescription>
          </CardHeader>
          <CardContent className="pb-5">
            <table className="w-full text-sm">
              <thead className="text-xs text-muted-foreground">
                <tr className="text-left">
                  <th className="py-1 font-medium">Model</th>
                  <th className="text-right font-medium">AUC</th>
                  <th className="text-right font-medium">PR-AUC</th>
                  <th className="text-right font-medium">Brier</th>
                </tr>
              </thead>
              <tbody className="tabular-nums">
                {[...report.candidates]
                  .sort((a, b) => (b.roc_auc ?? 0) - (a.roc_auc ?? 0))
                  .map((c) => (
                    <tr key={c.model} className={`border-t ${c.model === report.model_name ? "font-semibold" : ""}`}>
                      <td className="py-1.5">{c.model.replace(/_/g, " ")}</td>
                      <td className="text-right">{c.roc_auc?.toFixed(4)}</td>
                      <td className="text-right">{c.pr_auc?.toFixed(4)}</td>
                      <td className="text-right">{c.brier_score?.toFixed(4)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Metric({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <Card>
      <CardContent className="py-5">
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
        <p className="text-2xl font-semibold tabular-nums tracking-tight">{value}</p>
        <p className="text-xs text-muted-foreground">{hint}</p>
      </CardContent>
    </Card>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between border-b pb-2 last:border-0 last:pb-0">
      <span className="text-muted-foreground">{k}</span>
      <span className="font-medium tabular-nums">{v}</span>
    </div>
  );
}
