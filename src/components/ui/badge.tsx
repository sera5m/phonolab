import type { HTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium tabular-nums",
  {
    variants: {
      variant: {
        default: "bg-surface-2 text-muted shadow-[var(--shadow-border)]",
        accent: "bg-accent/15 text-accent",
        steel: "bg-steel/20 text-fg",
        warn: "bg-warn/15 text-warn",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export function Badge({
  className,
  variant,
  ...props
}: HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
