"use client";
import * as React from "react";
import { useRouter } from "next/navigation";
import { LayoutDashboard, Moon, Sun, LogOut } from "lucide-react";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from "@/components/ui/command";
import { useUiStore } from "@/store/ui-store";
import { NAV_SECTIONS } from "@/config/nav";
import { useTheme } from "next-themes";
import { useLogout } from "@/hooks/use-auth";

export function CommandPalette() {
  const open = useUiStore((s) => s.commandPaletteOpen);
  const setOpen = useUiStore((s) => s.setCommandPaletteOpen);
  const router = useRouter();
  const { setTheme } = useTheme();
  const logout = useLogout();

  React.useEffect(() => {
    function handler(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen(!open);
      }
    }
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, setOpen]);

  const runAndClose = (fn: () => void) => {
    fn();
    setOpen(false);
  };

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput placeholder="Search pages, actions..." />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>
        <CommandGroup heading="Navigate">
          {NAV_SECTIONS.flatMap((s) => s.items)
            .filter((i) => !i.comingSoon)
            .map((item) => (
              <CommandItem key={item.href} onSelect={() => runAndClose(() => router.push(item.href))}>
                <LayoutDashboard />
                {item.label}
              </CommandItem>
            ))}
        </CommandGroup>
        <CommandGroup heading="Actions">
          <CommandItem onSelect={() => runAndClose(() => setTheme("dark"))}>
            <Moon /> Switch to dark theme
          </CommandItem>
          <CommandItem onSelect={() => runAndClose(() => setTheme("light"))}>
            <Sun /> Switch to light theme
          </CommandItem>
          <CommandItem onSelect={() => runAndClose(() => logout())}>
            <LogOut /> Sign out
            <CommandShortcut>⌘Q</CommandShortcut>
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
