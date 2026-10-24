"use client";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { PredictionDistribution } from "@/lib/types";

export function PredictionDistributionChart({
  data,
  loading,
}: {
  data?: PredictionDistribution;
  loading?: boolean;
}) {
  const chartData =
    data?.histogram.map((bin) => ({
      range: `${Math.round(bin.bin_start * 100)}–${Math.round(bin.bin_end * 100)}%`,
      count: bin.count,
    })) ?? [];

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>Predicted Probability Distribution</CardTitle>
        <CardDescription>How predicted default probabilities are spread across recent applications</CardDescription>
      </CardHeader>
      <CardContent className="pb-5">
        {loading ? (
          <Skeleton className="h-64 w-full" />
        ) : !data || data.n === 0 ? (
          <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
            No predictions recorded yet.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={256}>
            <BarChart data={chartData} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" vertical={false} />
              <XAxis dataKey="range" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} allowDecimals={false} />
              <Tooltip
                cursor={{ fill: "var(--color-muted)" }}
                contentStyle={{
                  background: "var(--color-popover)",
                  border: "1px solid var(--color-border)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
              <Bar dataKey="count" fill="var(--color-chart-1)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
