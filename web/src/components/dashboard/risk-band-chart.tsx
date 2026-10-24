"use client";
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { RISK_BAND_LABELS, type PredictionDistribution } from "@/lib/types";

const BAND_COLORS: Record<string, string> = {
  very_low: "var(--color-chart-2)",
  low: "var(--color-success)",
  medium: "var(--color-warning)",
  high: "var(--color-chart-4)",
  very_high: "var(--color-danger)",
};

export function RiskBandChart({ data, loading }: { data?: PredictionDistribution; loading?: boolean }) {
  const chartData = Object.entries(data?.risk_band_counts ?? {}).map(([band, count]) => ({
    name: RISK_BAND_LABELS[band] ?? band,
    key: band,
    value: count,
  }));

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>Risk Band Breakdown</CardTitle>
        <CardDescription>Share of applications in each risk tier</CardDescription>
      </CardHeader>
      <CardContent className="pb-5">
        {loading ? (
          <Skeleton className="h-64 w-full" />
        ) : chartData.length === 0 ? (
          <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
            No predictions recorded yet.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={256}>
            <PieChart>
              <Pie data={chartData} dataKey="value" nameKey="name" innerRadius={55} outerRadius={90} paddingAngle={2}>
                {chartData.map((entry) => (
                  <Cell key={entry.key} fill={BAND_COLORS[entry.key] ?? "var(--color-chart-5)"} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: "var(--color-popover)",
                  border: "1px solid var(--color-border)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
