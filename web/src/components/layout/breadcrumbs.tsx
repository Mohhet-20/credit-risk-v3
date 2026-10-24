"use client";
import { usePathname } from "next/navigation";
import { NAV_SECTIONS } from "@/config/nav";

export function Breadcrumbs() {
  const pathname = usePathname();
  const current = NAV_SECTIONS.flatMap((s) => s.items).find((i) => i.href === pathname);

  return (
    <div className="flex items-center gap-1.5 text-sm">
      <span className="text-muted-foreground">Platform</span>
      <span className="text-muted-foreground">/</span>
      <span className="font-medium">{current?.label ?? "Overview"}</span>
    </div>
  );
}
