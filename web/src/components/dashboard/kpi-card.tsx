"use client";
import * as React from "react";
import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface KpiCardProps {
  label: string;
  value: string;
  icon: LucideIcon;
  loading?: boolean;
  hint?: string;
  trend?: { value: string; direction: "up" | "down"; positive?: boolean };
  tone?: "default" | "success" | "warning" | "danger";
}

const toneStyles: Record<NonNullable<KpiCardProps["tone"]>, string> = {
  default: "bg-primary/10 text-primary",
  success: "bg-success/10 text-success",
  warning: "bg-warning/10 text-warning",
  danger: "bg-danger/10 text-danger",
};

export function KpiCard({ label, value, icon: Icon, loading, hint, trend, tone = "default" }: KpiCardProps) {
  return (
    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
      <Card className="h-full">
        <CardContent className="pt-5 pb-5">
          <div className="flex items-start justify-between">
            <div className="flex flex-col gap-1">
              <span className="text-xs font-medium text-muted-foreground">{label}</span>
              {loading ? (
                <Skeleton className="h-7 w-20" />
              ) : (
                <span className="text-2xl font-semibold tracking-tight tabular-nums">{value}</span>
              )}
              {hint && !loading && <span className="text-xs text-muted-foreground">{hint}</span>}
            </div>
            <div className={cn("flex size-9 shrink-0 items-center justify-center rounded-lg", toneStyles[tone])}>
              <Icon className="size-4.5" />
            </div>
          </div>
          {trend && !loading && (
            <div
              className={cn(
                "mt-3 flex items-center gap-1 text-xs font-medium",
                trend.positive === false ? "text-danger" : "text-success"
              )}
            >
              {trend.direction === "up" ? <ArrowUpRight className="size-3.5" /> : <ArrowDownRight className="size-3.5" />}
              {trend.value}
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}
