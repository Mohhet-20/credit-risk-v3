"use client";
import axios, { AxiosError, type AxiosInstance } from "axios";
import { useAuthStore } from "@/store/auth-store";

// Design decision: mirrors the Python Streamlit frontend's
// `frontend/utils/api_client.py` -- a thin, typed wrapper over the FastAPI
// backend, with the JWT read from client state (here Zustand+persist,
// there `st.session_state`) rather than the app importing
// `training`/`models`/`explainability` directly. Both frontends are equally
// "just a client" of the versioned `/api/v1` contract.
// "/" means same origin: the hosted demo proxies /api/* to the API (web/vercel.json), so requests stay relative.
export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/+$/, "");
export const API_PREFIX = "/api/v1";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

function createClient(): AxiosInstance {
  const client = axios.create({ baseURL: API_BASE_URL, timeout: 30_000 });

  client.interceptors.request.use((config) => {
    const token = useAuthStore.getState().accessToken;
    if (token) {
      config.headers = config.headers ?? {};
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  });

  client.interceptors.response.use(
    (response) => response,
    (error: AxiosError<{ detail?: string }>) => {
      const status = error.response?.status ?? 0;
      const detail = error.response?.data?.detail ?? error.message ?? "Request failed";
      if (status === 401) {
        useAuthStore.getState().clear();
      }
      return Promise.reject(new ApiError(status, typeof detail === "string" ? detail : JSON.stringify(detail)));
    }
  );

  return client;
}

export const apiClient = createClient();

export async function apiGet<T>(path: string, params?: Record<string, unknown>): Promise<T> {
  const { data } = await apiClient.get<T>(`${API_PREFIX}${path}`, { params });
  return data;
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const { data } = await apiClient.post<T>(`${API_PREFIX}${path}`, body);
  return data;
}

export async function apiGetRaw<T>(path: string, params?: Record<string, unknown>): Promise<T> {
  // For endpoints outside the /api/v1 prefix, e.g. /health.
  const { data } = await apiClient.get<T>(path, { params });
  return data;
}
