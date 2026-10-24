import type { LucideIcon } from "lucide-react";
import { LayoutDashboard, Scale, Target } from "lucide-react";

export interface NavItem {
  label: string;
  href: string;
  icon: LucideIcon;
  /** Pages not yet built in this frontend pass -- shown, disabled, badged "Soon". */
  comingSoon?: boolean;
}

export interface NavSection {
  title: string;
  items: NavItem[];
}

export const NAV_SECTIONS: NavSection[] = [
  {
    title: "Overview",
    items: [{ label: "Executive Dashboard", href: "/", icon: LayoutDashboard }],
  },
  {
    title: "Decisioning",
    items: [{ label: "Score Application", href: "/predict", icon: Target }],
  },
  {
    title: "Model Risk",
    items: [{ label: "Model Validation", href: "/validation", icon: Scale }],
  },
];
