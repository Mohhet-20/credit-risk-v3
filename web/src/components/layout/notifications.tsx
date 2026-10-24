"use client";
import { Bell } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useHealth } from "@/hooks/use-dashboard";

export function Notifications() {
  const { data: health } = useHealth();
  const hasAlert = health?.status !== "ok";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" className="relative" aria-label="Notifications">
          <Bell className="size-4" />
          {hasAlert && <span className="absolute top-1.5 right-1.5 size-1.5 rounded-full bg-danger" />}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-80">
        <DropdownMenuLabel>Notifications</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {health?.status === "ok" ? (
          <DropdownMenuItem disabled className="flex-col items-start gap-0.5 opacity-100">
            <span className="font-medium">All systems operational</span>
            <span className="text-xs text-muted-foreground">
              Champion model: {health.champion_model ?? "none"} ({health.champion_version?.slice(0, 24) ?? "n/a"})
            </span>
          </DropdownMenuItem>
        ) : (
          <DropdownMenuItem disabled className="flex-col items-start gap-0.5 opacity-100">
            <span className="font-medium text-danger">API unreachable</span>
            <span className="text-xs text-muted-foreground">Check the backend service status.</span>
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
