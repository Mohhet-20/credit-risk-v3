import { ShieldHalf } from "lucide-react";

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative flex min-h-dvh items-center justify-center overflow-hidden bg-background px-4">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-40"
        style={{
          backgroundImage:
            "radial-gradient(circle at 20% 20%, var(--color-primary) 0%, transparent 35%), radial-gradient(circle at 80% 80%, var(--color-chart-2) 0%, transparent 35%)",
          filter: "blur(80px)",
        }}
      />
      <div className="relative z-10 w-full max-w-sm">
        <div className="mb-8 flex flex-col items-center gap-2 text-center">
          <div className="flex size-11 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-lg">
            <ShieldHalf className="size-5" />
          </div>
          <h1 className="text-lg font-semibold tracking-tight">Credit Risk Intelligence Platform</h1>
          <p className="text-sm text-muted-foreground">Enterprise credit decisioning, explained.</p>
        </div>
        {children}
      </div>
    </div>
  );
}
