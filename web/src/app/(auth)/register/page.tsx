"use client";
import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Loader2, UserPlus } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useRegister } from "@/hooks/use-auth";
import { ApiError } from "@/lib/api-client";

const ROLES = [
  { value: "viewer", label: "Viewer -- read-only" },
  { value: "loan_officer", label: "Loan Officer -- score & explain" },
  { value: "risk_analyst", label: "Risk Analyst -- monitoring & governance" },
  { value: "admin", label: "Admin -- full access" },
];

export default function RegisterPage() {
  const router = useRouter();
  const [username, setUsername] = React.useState("");
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [role, setRole] = React.useState("loan_officer");
  const register = useRegister();

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    register.mutate(
      { username, email, password, role },
      {
        onSuccess: () => {
          toast.success("Account created", { description: "You can now sign in." });
          router.push("/login");
        },
        onError: (err) => {
          const message = err instanceof ApiError ? err.message : "Something went wrong.";
          toast.error("Registration failed", { description: message });
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
              <Input id="username" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="jane.doe" required />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="jane@bank.com" required />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="At least 8 characters"
                minLength={8}
                required
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="role">Role</Label>
              <Select value={role} onValueChange={setRole}>
                <SelectTrigger id="role">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ROLES.map((r) => (
                    <SelectItem key={r.value} value={r.value}>
                      {r.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button type="submit" className="mt-1 w-full" disabled={register.isPending}>
              {register.isPending ? <Loader2 className="animate-spin" /> : <UserPlus />}
              Create account
            </Button>
          </form>
        </CardContent>
        <CardFooter className="flex-col gap-2 border-t pt-4">
          <p className="text-center text-sm text-muted-foreground">
            Already have an account?{" "}
            <Link href="/login" className="font-medium text-primary hover:underline">
              Sign in
            </Link>
          </p>
        </CardFooter>
      </Card>
    </motion.div>
  );
}
