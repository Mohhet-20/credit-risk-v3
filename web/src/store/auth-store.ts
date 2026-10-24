"use client";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { UserOut } from "@/lib/types";

interface AuthState {
  accessToken: string | null;
  refreshToken: string | null;
  user: UserOut | null;
  setSession: (accessToken: string, refreshToken: string, user: UserOut) => void;
  clear: () => void;
}

// Design decision: token + user identity are persisted to localStorage via
// zustand's `persist` middleware, mirroring the Streamlit frontend's use of
// `st.session_state` for the same purpose (see frontend/utils/api_client.py
// in the Python app) -- both are "keep the session alive across a page
// reload without a server-side session store" for a SPA/MPA hybrid that has
// no cookie-based session of its own; the JWT itself is what the FastAPI
// backend actually trusts.
export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      accessToken: null,
      refreshToken: null,
      user: null,
      setSession: (accessToken, refreshToken, user) => set({ accessToken, refreshToken, user }),
      clear: () => set({ accessToken: null, refreshToken: null, user: null }),
    }),
    { name: "credit-risk-auth" }
  )
);
