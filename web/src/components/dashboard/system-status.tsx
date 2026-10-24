"use client";
import { CheckCircle2, Cpu, Gauge, XCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import type { HealthResponse, LatencyStats } from "@/lib/types";

export function SystemStatus({
  health,
  latency,
  loading,
}: {
  health?: HealthResponse;
  latency?: LatencyStats;
  loading?: boolean;
}) {
  const isUp = health?.status === "ok";

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>System Status</CardTitle>
        <CardDescription>API health and the currently serving model</CardDescription>
      </CardHeader>
      <CardContent className="pb-5">
        {loading ? (
          <Skeleton className="h-40 w-full" />
        ) : (
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-3">
              {isUp ? (
                <CheckCircle2 className="size-5 text-success" />
              ) : (
                <XCircle className="size-5 text-danger" />
              )}
              <div>
                <p className="text-sm font-medium">{isUp ? "All systems operational" : "API unreachable"}</p>
                <p className="text-xs text-muted-foreground">FastAPI backend health check</p>
              </div>
            </div>

            <Separator />

            <div className="flex items-center gap-3">
              <Cpu className="size-5 text-primary" />
              <div className="min-w-0">
                <p className="truncate text-sm font-medium">{health?.champion_model ?? "No champion model"}</p>
                <p className="truncate text-xs text-muted-foreground">
                  {health?.champion_version ?? "Promote a model to see it here"}
                </p>
              </div>
            </div>

            <Separator />

            <div className="flex items-center gap-3">
              <Gauge className="size-5 text-primary" />
              <div>
                <p className="text-sm font-medium">
                  {latency && latency.n > 0 ? `${latency.p50_ms?.toFixed(1)} ms p50` : "No latency data yet"}
                </p>
                <p className="text-xs text-muted-foreground">
                  {latency && latency.n > 0
                    ? `p95 ${latency.p95_ms?.toFixed(1)} ms · p99 ${latency.p99_ms?.toFixed(1)} ms`
                    : "Inference latency percentiles"}
                </p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
