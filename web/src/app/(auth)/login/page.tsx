"use client";
import * as React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Loader2, LogIn, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { useGuestSession, useLogin } from "@/hooks/use-auth";
import { ApiError } from "@/lib/api-client";

// Inlined at build time; the hosted demo sets it in frontend hosting (web/vercel.json).
const GUEST_ACCESS = process.env.NEXT_PUBLIC_GUEST_ACCESS === "true";

export default function LoginPage() {
  const [username, setUsername] = React.useState("");
  const [password, setPassword] = React.useState("");
  const login = useLogin();
  const guest = useGuestSession();

  function exploreDemo() {
    guest.mutate(undefined, {
      onError: (err) => {
        const message = err instanceof ApiError ? err.message : "Something went wrong.";
        toast.error("Could not start a demo session", { description: message });
      },
    });
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    login.mutate(
      { username, password },
      {
        onError: (err) => {
          const message = err instanceof ApiError ? err.message : "Something went wrong.";
          toast.error("Sign in failed", { description: message });
        },
      }
    );
  }

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
      <Card className="glass shadow-xl">
        <CardContent className="pt-1">
          {GUEST_ACCESS && (
            <div className="mb-5 flex flex-col gap-3">
              <Button type="button" className="w-full" onClick={exploreDemo} disabled={guest.isPending}>
                {guest.isPending ? <Loader2 className="animate-spin" /> : <Sparkles />}
                Explore the live demo
              </Button>
              <p className="text-xs leading-relaxed text-muted-foreground">
                Score applications, see the decline reasons a lender must give, and read the model validation report,
                as a risk analyst. Guest sessions can&apos;t administer the platform; the demo resets when the server
                restarts.
              </p>
              <div className="flex items-center gap-3 text-xs text-muted-foreground" aria-hidden>
                <span className="h-px flex-1 bg-border" />
                or
                <span className="h-px flex-1 bg-border" />
              </div>
            </div>
          )}
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="jane.doe"
                required
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between">
                <Label htmlFor="password">Password</Label>
                <span className="text-xs text-muted-foreground cursor-not-allowed" title="Coming soon">
                  Forgot password?
                </span>
              </div>
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
              />
            </div>
            <Button type="submit" className="mt-1 w-full" disabled={login.isPending}>
              {login.isPending ? <Loader2 className="animate-spin" /> : <LogIn />}
              Sign in
            </Button>
          </form>
        </CardContent>
        {!GUEST_ACCESS && (
          <CardFooter className="flex-col gap-2 border-t pt-4">
            <p className="text-center text-sm text-muted-foreground">
              No account?{" "}
              <Link href="/register" className="font-medium text-primary hover:underline">
                Create one
              </Link>
            </p>
          </CardFooter>
        )}
      </Card>
    </motion.div>
  );
}
