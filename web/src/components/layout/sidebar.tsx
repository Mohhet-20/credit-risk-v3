"use client";
import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { ChevronsLeft, ShieldHalf } from "lucide-react";
import { cn } from "@/lib/utils";
import { NAV_SECTIONS } from "@/config/nav";
import { useUiStore } from "@/store/ui-store";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export function Sidebar({ className }: { className?: string }) {
  const pathname = usePathname();
  const collapsed = useUiStore((s) => s.sidebarCollapsed);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);

  return (
    <motion.aside
      animate={{ width: collapsed ? 72 : 248 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className={cn(
        "hidden md:flex h-full flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground overflow-hidden",
        className
      )}
    >
      <div className="flex h-14 items-center gap-2 border-b border-sidebar-border px-4">
        <div className="flex size-7 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground">
          <ShieldHalf className="size-4" />
        </div>
        {!collapsed && (
          <span className="truncate text-sm font-semibold tracking-tight">Credit Risk Platform</span>
        )}
      </div>

      <nav className="flex-1 space-y-5 overflow-y-auto scrollbar-thin px-3 py-4">
        {NAV_SECTIONS.map((section) => (
          <div key={section.title}>
            {!collapsed && (
              <p className="mb-1.5 px-2 text-[11px] font-medium tracking-wider text-muted-foreground uppercase">
                {section.title}
              </p>
            )}
            <div className="space-y-0.5">
              {section.items.map((item) => {
                const active = pathname === item.href;
                const Icon = item.icon;
                const linkContent = (
                  <span
                    className={cn(
                      "group flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm font-medium transition-colors",
                      active
                        ? "bg-sidebar-accent text-sidebar-accent-foreground"
                        : "text-sidebar-foreground/70 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground",
                      item.comingSoon && "opacity-60"
                    )}
                  >
                    <Icon className="size-4 shrink-0" />
                    {!collapsed && <span className="truncate">{item.label}</span>}
                    {!collapsed && item.comingSoon && (
                      <Badge variant="secondary" className="ml-auto text-[10px] px-1.5 py-0">
                        Soon
                      </Badge>
                    )}
                  </span>
                );

                const el = item.comingSoon ? (
                  <div key={item.href} aria-disabled className="cursor-not-allowed">
                    {linkContent}
                  </div>
                ) : (
                  <Link key={item.href} href={item.href}>
                    {linkContent}
                  </Link>
                );

                return collapsed ? (
                  <Tooltip key={item.href}>
                    <TooltipTrigger asChild>{el}</TooltipTrigger>
                    <TooltipContent side="right">{item.label}{item.comingSoon ? " (Soon)" : ""}</TooltipContent>
                  </Tooltip>
                ) : (
                  el
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="border-t border-sidebar-border p-2">
        <button
          onClick={toggleSidebar}
          className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-sm text-sidebar-foreground/70 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground"
        >
          <ChevronsLeft className={cn("size-4 transition-transform", collapsed && "rotate-180")} />
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </motion.aside>
  );
}
