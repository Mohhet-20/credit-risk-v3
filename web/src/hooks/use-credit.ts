"use client";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiGet, apiPost } from "@/lib/api-client";
import { useAuthStore } from "@/store/auth-store";
import type { CreditApplication, PredictionResponse, ValidationReport } from "@/lib/types";

export function useScoreApplication() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (application: CreditApplication) => apiPost<PredictionResponse>("/predict", application),
    onSuccess: () => {
      // New prediction changes the dashboard's volume, distribution and history.
      queryClient.invalidateQueries({ queryKey: ["prediction-history"] });
      queryClient.invalidateQueries({ queryKey: ["prediction-distribution"] });
      queryClient.invalidateQueries({ queryKey: ["monitoring-stats"] });
    },
  });
}

export function useValidationReport() {
  const isAuthed = useAuthStore((s) => Boolean(s.accessToken));
  return useQuery({
    queryKey: ["validation-report"],
    queryFn: () => apiGet<ValidationReport>("/models/validation-report"),
    enabled: isAuthed,
    retry: false,
  });
}
