"use client";
import { useQuery } from "@tanstack/react-query";
import { apiGet, apiGetRaw } from "@/lib/api-client";
import { useAuthStore } from "@/store/auth-store";
import type {
  HealthResponse,
  LatencyStats,
  ModelVersionOut,
  MonitoringStats,
  PredictionDistribution,
  PredictionHistoryItem,
} from "@/lib/types";

// Design decision: every hook here is a thin `useQuery` wrapper, matching
// the same "the frontend is just an HTTP client" principle the Python
// Streamlit app followed (see `frontend/utils/api_client.py`). React Query
// handles caching/refetch/loading state, so components stay declarative.

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => apiGetRaw<HealthResponse>("/health"),
    refetchInterval: 30_000,
  });
}

export function useMonitoringStats(days = 30) {
  const isAuthed = useAuthStore((s) => Boolean(s.accessToken));
  return useQuery({
    queryKey: ["monitoring-stats", days],
    queryFn: () => apiGet<MonitoringStats>("/monitoring/stats", { days }),
    enabled: isAuthed,
    refetchInterval: 30_000,
  });
}

export function usePredictionDistribution(days = 30) {
  const isAuthed = useAuthStore((s) => Boolean(s.accessToken));
  return useQuery({
    queryKey: ["prediction-distribution", days],
    queryFn: () => apiGet<PredictionDistribution>("/monitoring/prediction-distribution", { days }),
    enabled: isAuthed,
    refetchInterval: 30_000,
  });
}

export function useLatencyStats() {
  const isAuthed = useAuthStore((s) => Boolean(s.accessToken));
  return useQuery({
    queryKey: ["latency-stats"],
    queryFn: () => apiGet<LatencyStats>("/monitoring/latency"),
    enabled: isAuthed,
    refetchInterval: 30_000,
  });
}

export function useModelVersions() {
  const isAuthed = useAuthStore((s) => Boolean(s.accessToken));
  return useQuery({
    queryKey: ["model-versions"],
    queryFn: () => apiGet<ModelVersionOut[]>("/models/versions"),
    enabled: isAuthed,
  });
}

export function usePredictionHistory(limit = 8) {
  const isAuthed = useAuthStore((s) => Boolean(s.accessToken));
  return useQuery({
    queryKey: ["prediction-history", limit],
    queryFn: () => apiGet<PredictionHistoryItem[]>("/predict/history", { limit }),
    enabled: isAuthed,
    refetchInterval: 20_000,
  });
}
