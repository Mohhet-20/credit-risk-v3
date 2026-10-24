"use client";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Breadcrumbs } from "@/components/layout/breadcrumbs";
import { Notifications } from "@/components/layout/notifications";
import { ThemeToggle } from "@/components/layout/theme-toggle";
import { UserMenu } from "@/components/layout/user-menu";
import { MobileSidebar } from "@/components/layout/mobile-sidebar";
import { useUiStore } from "@/store/ui-store";

export function Topbar() {
  const setCommandPaletteOpen = useUiStore((s) => s.setCommandPaletteOpen);

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b bg-background/80 backdrop-blur px-4">
      <MobileSidebar />
      <Breadcrumbs />

      <button
        onClick={() => setCommandPaletteOpen(true)}
        className="ml-4 hidden flex-1 max-w-sm items-center gap-2 rounded-md border bg-muted/40 px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted sm:flex"
      >
        <Search className="size-3.5" />
        <span>Search or jump to...</span>
        <kbd className="ml-auto rounded border bg-background px-1.5 py-0.5 text-[10px] font-medium">⌘K</kbd>
      </button>

      <div className="ml-auto flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon"
          className="sm:hidden"
          onClick={() => setCommandPaletteOpen(true)}
          aria-label="Search"
        >
          <Search className="size-4" />
        </Button>
        <Notifications />
        <ThemeToggle />
        <div className="mx-1 h-6 w-px bg-border" />
        <UserMenu />
      </div>
    </header>
  );
}
