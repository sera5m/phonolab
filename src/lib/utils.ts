import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

export function formatHz(hz: number) {
  if (!Number.isFinite(hz) || hz <= 0) return "—";
  if (hz >= 1000) return `${(hz / 1000).toFixed(2)} kHz`;
  return `${Math.round(hz)} Hz`;
}

export function formatDb(db: number) {
  if (!Number.isFinite(db)) return "—";
  return `${db.toFixed(1)} dB`;
}

export function formatMs(seconds: number) {
  return `${Math.round(seconds * 1000)} ms`;
}

export function formatPct(n: number) {
  return `${Math.round(clamp(n, 0, 1) * 100)}%`;
}
