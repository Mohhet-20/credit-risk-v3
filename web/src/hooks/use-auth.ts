"use client";
import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { API_PREFIX, apiClient, apiPost, ApiError } from "@/lib/api-client";
import { useAuthStore } from "@/store/auth-store";
import type { LoginResponse, UserOut } from "@/lib/types";

interface LoginInput {
  username: string;
  password: string;
}

interface RegisterInput {
  username: string;
  email: string;
  password: string;
  role: string;
}

// Login returns tokens only; the profile (id, email, role) comes from
// `GET /auth/me` so the UI never trusts a client-side decode of the JWT.
// Accounts with MFA enabled get a challenge token instead of tokens; that
// second step is surfaced as an error here until the TOTP screen exists.
export function useLogin() {
  const router = useRouter();
  const setSession = useAuthStore((s) => s.setSession);

  return useMutation({
    mutationFn: async (input: LoginInput) => {
      const login = await apiPost<LoginResponse>("/auth/login", input);
      if (login.mfa_required || !login.access_token || !login.refresh_token) {
        throw new ApiError(401, "This account requires multi-factor authentication.");
      }
      const { data: user } = await apiClient.get<UserOut>(`${API_PREFIX}/auth/me`, {
        headers: { Authorization: `Bearer ${login.access_token}` },
      });
      return { tokens: login, user };
    },
    onSuccess: ({ tokens, user }) => {
      setSession(tokens.access_token!, tokens.refresh_token!, user);
      router.push("/");
    },
  });
}

export function useRegister() {
  return useMutation({
    mutationFn: (input: RegisterInput) => apiPost<UserOut>("/auth/register", input),
  });
}

export function useLogout() {
  const router = useRouter();
  const clear = useAuthStore((s) => s.clear);
  return () => {
    clear();
    router.push("/login");
  };
}

export type { LoginInput, RegisterInput };
export { ApiError };
