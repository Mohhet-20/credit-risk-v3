"use client";
import { formatDistanceToNow } from "date-fns";
import { Activity } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { RISK_BAND_LABELS, type PredictionHistoryItem } from "@/lib/types";

const DECISION_VARIANT: Record<string, "success" | "danger" | "warning"> = {
  approved: "success",
  rejected: "danger",
  manual_review: "warning",
};

export function RecentActivity({ items, loading }: { items?: PredictionHistoryItem[]; loading?: boolean }) {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>Recent Activity</CardTitle>
        <CardDescription>Latest scored applications across the platform</CardDescription>
      </CardHeader>
      <CardContent className="pb-5">
        {loading ? (
          <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-10 w-full" />
            ))}
          </div>
        ) : !items || items.length === 0 ? (
          <div className="flex h-40 flex-col items-center justify-center gap-2 text-sm text-muted-foreground">
            <Activity className="size-6 opacity-40" />
            No predictions yet -- score an applicant to see activity here.
          </div>
        ) : (
          <ul className="divide-y divide-border">
            {items.map((item) => (
              <li key={item.prediction_id} className="flex items-center justify-between gap-3 py-2.5 text-sm">
                <div className="flex min-w-0 flex-col">
                  <span className="truncate font-medium">{item.customer_id ?? "Anonymous applicant"}</span>
                  <span className="text-xs text-muted-foreground">
                    {item.model_name} &middot; {RISK_BAND_LABELS[item.risk_band] ?? item.risk_band} risk
                  </span>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <span className="text-xs tabular-nums text-muted-foreground">
                    {(item.predicted_probability * 100).toFixed(1)}%
                  </span>
                  <Badge variant={DECISION_VARIANT[item.decision] ?? "secondary"} className="capitalize">
                    {item.decision.replace("_", " ")}
                  </Badge>
                  <span className="hidden w-20 shrink-0 text-right text-xs text-muted-foreground sm:block">
                    {item.created_at ? formatDistanceToNow(new Date(item.created_at), { addSuffix: true }) : "--"}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
