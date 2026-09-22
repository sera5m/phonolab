import { useEffect, useRef, type MouseEvent } from "react";
import type { AnalysisResult, PhonemeHit } from "@/lib/phonetics/types";
import { cn } from "@/lib/utils";

type Props = {
  analysis: AnalysisResult;
  selected: PhonemeHit | null;
  expanded: boolean;
  playhead: number | null;
  onSelectTime: (t: number) => void;
};

export function Spectrogram({ analysis, selected, expanded, playhead, onSelectTime }: Props) {
  const specRef = useRef<HTMLCanvasElement>(null);
  const waveRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = specRef.current;
    if (!canvas) return;
    const { data, rows, cols } = analysis.spectrogram;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cssW = canvas.clientWidth || 640;
    const cssH = canvas.clientHeight || 160;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    const img = ctx.createImageData(cols, rows);
    for (let i = 0; i < data.length; i++) {
      const v = data[i] ?? 0;
      const c = heat(v);
      img.data[i * 4] = c[0];
      img.data[i * 4 + 1] = c[1];
      img.data[i * 4 + 2] = c[2];
      img.data[i * 4 + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = cols;
    tmp.height = rows;
    tmp.getContext("2d")?.putImageData(img, 0, 0);
    ctx.fillStyle = "#09090b";
    ctx.fillRect(0, 0, cssW, cssH);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tmp, 0, 0, cssW, cssH);

    const dur = analysis.duration || 1;
    const nyquist = (analysis.spectrogram.binHz * analysis.spectrogram.rows) / 2 || 8000;

    ctx.globalAlpha = 0.85;
    for (const p of analysis.formantTrack) {
      const x = (p.t / dur) * cssW;
      const y = (f: number) => cssH - (f / nyquist) * cssH * 0.92;
      if (p.f1 > 80) dot(ctx, x, y(p.f1), "#8fd4c4", 1.4);
      if (p.f2 > 80) dot(ctx, x, y(p.f2), "#c4a574", 1.4);
      if (p.f3 > 80) dot(ctx, x, y(p.f3), "#6b7c8a", 1.2);
    }
    ctx.globalAlpha = 1;

    if (selected) {
      const x0 = (selected.start / dur) * cssW;
      const x1 = (selected.end / dur) * cssW;
      ctx.fillStyle = "rgba(143, 212, 196, 0.12)";
      ctx.fillRect(x0, 0, Math.max(2, x1 - x0), cssH);
      ctx.strokeStyle = "rgba(143, 212, 196, 0.7)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x0 + 0.5, 0.5, Math.max(2, x1 - x0), cssH - 1);
    }
    if (playhead != null) {
      const x = (playhead / dur) * cssW;
      ctx.strokeStyle = "#f1f0ea";
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, cssH);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }, [analysis, selected, expanded, playhead]);

  useEffect(() => {
    const canvas = waveRef.current;
    if (!canvas) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cssW = canvas.clientWidth || 640;
    const cssH = canvas.clientHeight || 48;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssW, cssH);
    const wave = analysis.waveform;
    const mid = cssH / 2;
    ctx.strokeStyle = "#8fd4c4";
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < wave.length; i++) {
      const x = (i / (wave.length - 1)) * cssW;
      const h = (wave[i] ?? 0) * (cssH * 0.46);
      ctx.moveTo(x, mid - h);
      ctx.lineTo(x, mid + h);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    if (selected) {
      const dur = analysis.duration || 1;
      const x0 = (selected.start / dur) * cssW;
      const x1 = (selected.end / dur) * cssW;
      ctx.fillStyle = "rgba(143, 212, 196, 0.14)";
      ctx.fillRect(x0, 0, Math.max(2, x1 - x0), cssH);
    }
  }, [analysis, selected]);

  function handleClick(e: MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const t = ((e.clientX - rect.left) / rect.width) * analysis.duration;
    onSelectTime(t);
  }

  return (
    <div className="flex flex-col gap-2">
      <canvas
        ref={waveRef}
        className="h-12 w-full cursor-crosshair rounded-sm bg-surface-2"
        onClick={handleClick}
        aria-label="Waveform"
      />
      <canvas
        ref={specRef}
        className={cn(
          "w-full cursor-crosshair rounded-md bg-surface-2 transition-[height] duration-250 ease-[cubic-bezier(0.22,1,0.36,1)]",
          expanded ? "h-72 md:h-80" : "h-36 md:h-44",
        )}
        onClick={handleClick}
        aria-label="Spectrogram"
      />
      <div className="flex items-center justify-between font-mono text-xs text-subtle">
        <span>0.00 s</span>
        <span className="flex items-center gap-3">
          <Legend swatch="#8fd4c4" label="F1" />
          <Legend swatch="#c4a574" label="F2" />
          <Legend swatch="#6b7c8a" label="F3" />
        </span>
        <span>{analysis.duration.toFixed(2)} s</span>
      </div>
    </div>
  );
}

function Legend({ swatch, label }: { swatch: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="size-1.5 rounded-full" style={{ background: swatch }} />
      {label}
    </span>
  );
}

function heat(v: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, v));
  if (t < 0.25) {
    const u = t / 0.25;
    return [9 + 17 * u, 9 + 65 * u, 11 + 57 * u];
  }
  if (t < 0.65) {
    const u = (t - 0.25) / 0.4;
    return [26 + 117 * u, 74 + 138 * u, 68 + 128 * u];
  }
  const u = (t - 0.65) / 0.35;
  return [143 + 98 * u, 212 + 28 * u, 196 + 38 * u];
}

function dot(ctx: CanvasRenderingContext2D, x: number, y: number, color: string, r: number) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}
