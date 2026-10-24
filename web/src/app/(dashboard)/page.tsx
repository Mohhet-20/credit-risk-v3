"use client";
import { Activity, AlertTriangle, CheckCircle2, Gauge, Target, TrendingDown, TrendingUp, Wifi } from "lucide-react";
import { PageHeader } from "@/components/dashboard/page-header";
import { KpiCard } from "@/components/dashboard/kpi-card";
import { PredictionDistributionChart } from "@/components/dashboard/prediction-distribution-chart";
import { RiskBandChart } from "@/components/dashboard/risk-band-chart";
import { RecentActivity } from "@/components/dashboard/recent-activity";
import { SystemStatus } from "@/components/dashboard/system-status";
import {
  useHealth,
  useLatencyStats,
  useModelVersions,
  useMonitoringStats,
  usePredictionDistribution,
  usePredictionHistory,
} from "@/hooks/use-dashboard";

export default function ExecutiveDashboardPage() {
  const { data: health, isLoading: healthLoading } = useHealth();
  const { data: stats, isLoading: statsLoading } = useMonitoringStats(30);
  const { data: distribution, isLoading: distributionLoading } = usePredictionDistribution(30);
  const { data: latency, isLoading: latencyLoading } = useLatencyStats();
  const { data: versions, isLoading: versionsLoading } = useModelVersions();
  const { data: history, isLoading: historyLoading } = usePredictionHistory(8);

  const n = distribution?.n ?? 0;
  const approved = distribution?.decision_counts?.approved ?? 0;
  const rejected = distribution?.decision_counts?.rejected ?? 0;
  const highRisk = (distribution?.risk_band_counts?.high ?? 0) + (distribution?.risk_band_counts?.very_high ?? 0);
  const champion = versions?.find((v) => v.is_champion);
  const rocAuc = champion?.test_metrics?.roc_auc;
  const gini = champion?.test_metrics?.gini;
  const ks = champion?.test_metrics?.ks;

  const pct = (num: number) => (n > 0 ? `${((num / n) * 100).toFixed(1)}%` : "--");

  return (
    <div>
      <PageHeader
        title="Executive Dashboard"
        description="Real-time overview of credit decisioning volume, risk, and model health"
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          label="Predictions (30d)"
          value={statsLoading ? "" : String(stats?.n_predictions ?? 0)}
          icon={Target}
          loading={statsLoading}
          hint="Scored applications"
        />
        <KpiCard
          label="Approval Rate"
          value={distributionLoading ? "" : pct(approved)}
          icon={CheckCircle2}
          loading={distributionLoading}
          tone="success"
          hint={`${approved} of ${n} applications`}
        />
        <KpiCard
          label="Rejection Rate"
          value={distributionLoading ? "" : pct(rejected)}
          icon={TrendingDown}
          loading={distributionLoading}
          tone="danger"
          hint={`${rejected} of ${n} applications`}
        />
        <KpiCard
          label="High-Risk Customers"
          value={distributionLoading ? "" : String(highRisk)}
          icon={AlertTriangle}
          loading={distributionLoading}
          tone="warning"
          hint={`${pct(highRisk)} of scored volume`}
        />
        <KpiCard
          label="Champion ROC-AUC (OOT)"
          value={versionsLoading ? "" : rocAuc != null ? rocAuc.toFixed(3) : "--"}
          icon={TrendingUp}
          loading={versionsLoading}
          hint={champion?.model_name?.replace(/_/g, " ") ?? "No champion promoted"}
        />
        <KpiCard
          label="Gini / KS (OOT)"
          value={versionsLoading ? "" : gini != null && ks != null ? `${gini.toFixed(2)} / ${ks.toFixed(2)}` : "--"}
          icon={Activity}
          loading={versionsLoading}
          hint="Out-of-time discrimination"
        />
        <KpiCard
          label="Avg. Latency"
          value={statsLoading ? "" : stats?.avg_latency_ms != null ? `${stats.avg_latency_ms.toFixed(0)} ms` : "--"}
          icon={Gauge}
          loading={statsLoading}
          hint="Per-prediction, last 30 days"
        />
        <KpiCard
          label="API Health"
          value={healthLoading ? "" : health?.status === "ok" ? "Operational" : "Degraded"}
          icon={Wifi}
          loading={healthLoading}
          tone={health?.status === "ok" ? "success" : "danger"}
          hint="FastAPI backend"
        />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <PredictionDistributionChart data={distribution} loading={distributionLoading} />
        <RiskBandChart data={distribution} loading={distributionLoading} />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <RecentActivity items={history} loading={historyLoading} />
        </div>
        <SystemStatus health={health} latency={latency} loading={healthLoading || latencyLoading} />
      </div>
    </div>
  );
}
