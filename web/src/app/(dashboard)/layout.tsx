"use client";
import * as React from "react";
import { useRouter } from "next/navigation";
import { Sidebar } from "@/components/layout/sidebar";
import { Topbar } from "@/components/layout/topbar";
import { CommandPalette } from "@/components/layout/command-palette";
import { useAuthStore } from "@/store/auth-store";
import { ShieldHalf } from "lucide-react";

function FullPageLoader() {
  return (
    <div className="flex h-dvh w-full items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-3">
        <div className="flex size-10 items-center justify-center rounded-lg bg-primary text-primary-foreground animate-pulse">
          <ShieldHalf className="size-5" />
        </div>
        <p className="text-sm text-muted-foreground">Loading platform...</p>
      </div>
    </div>
  );
}

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const accessToken = useAuthStore((s) => s.accessToken);
  const [hydrated, setHydrated] = React.useState(false);

  React.useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- standard zustand-persist SSR-safe hydration guard
    setHydrated(true);
  }, []);

  React.useEffect(() => {
    if (hydrated && !accessToken) router.replace("/login");
  }, [hydrated, accessToken, router]);

  if (!hydrated || !accessToken) return <FullPageLoader />;

  return (
    <div className="flex h-dvh overflow-hidden bg-background">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Topbar />
        <main className="flex-1 overflow-y-auto scrollbar-thin p-6">{children}</main>
      </div>
      <CommandPalette />
    </div>
  );
}
