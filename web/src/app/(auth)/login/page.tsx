"use client";
import * as React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Loader2, LogIn } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { useLogin } from "@/hooks/use-auth";
import { ApiError } from "@/lib/api-client";

export default function LoginPage() {
  const [username, setUsername] = React.useState("");
  const [password, setPassword] = React.useState("");
  const login = useLogin();

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
        <CardFooter className="flex-col gap-2 border-t pt-4">
          <p className="text-center text-sm text-muted-foreground">
            No account?{" "}
            <Link href="/register" className="font-medium text-primary hover:underline">
              Create one
            </Link>
          </p>
        </CardFooter>
      </Card>
    </motion.div>
  );
}
