import { Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { AttributeScores, PhonemeHit, Subphoneme } from "@/lib/phonetics/types";
import { cn, formatDb, formatHz, formatMs, formatPct } from "@/lib/utils";

const ATTR: { key: keyof AttributeScores; label: string; hint: string }[] = [
  { key: "brightness", label: "Brightness", hint: "Spectral centroid — energy sitting higher in the spectrum" },
  { key: "breathiness", label: "Breathiness", hint: "Noise vs. harmonic energy (inverse HNR)" },
  { key: "nasality", label: "Nasality", hint: "Low F1 and extra low-frequency pole" },
  { key: "roughness", label: "Roughness", hint: "Cycle-to-cycle jitter and shimmer" },
  { key: "tension", label: "Tension", hint: "Raised F0 with compressed F1" },
  { key: "resonance", label: "Resonance", hint: "How peaked the formant envelope is" },
];

type Props = {
  phoneme: PhonemeHit;
  onPlay: () => void;
};

export function PhonemeDetail({ phoneme: p, onPlay }: Props) {
  const g = p.gender;
  const needle = ((g.score + 1) / 2) * 100;
  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-subtle">Selected phoneme</p>
          <div className="mt-1 flex items-baseline gap-3">
            <span className="font-display text-5xl leading-none text-fg">/{p.ipa}/</span>
            <span className="font-mono text-sm text-muted">{p.arpabet}</span>
          </div>
          <p className="mt-2 text-sm text-muted">
            {p.example ? (
              <>
                as in <em className="text-fg">{p.example}</em>
              </>
            ) : (
              p.manner
            )}
            {p.word ? <span> · word “{p.word}”</span> : null}
            <span className="text-subtle"> · {formatMs(p.end - p.start)}</span>
          </p>
        </div>
        <Button variant="secondary" size="sm" onClick={onPlay} className="shrink-0">
          <Play className="ml-0.5" />
          Hear
        </Button>
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between text-xs">
          <span className="text-muted">Acoustic gender</span>
          <span className="font-mono text-fg">
            {g.label} · {formatPct(g.confidence)} conf.
          </span>
        </div>
        <div className="relative h-2 rounded-full bg-surface-2">
          <div
            className="absolute inset-y-0 left-0 rounded-full bg-steel/70"
            style={{ width: "50%" }}
          />
          <div
            className="absolute inset-y-0 right-0 rounded-full bg-accent/70"
            style={{ width: "50%" }}
          />
          <div
            className="absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-fg shadow-[var(--shadow-border)]"
            style={{ left: `${needle}%` }}
          />
        </div>
        <div className="mt-1 flex justify-between font-mono text-[11px] text-subtle">
          <span>masculine-coded</span>
          <span>{formatHz(g.f0Hz)}</span>
          <span>feminine-coded</span>
        </div>
        <ul className="mt-3 space-y-1 text-xs text-muted">
          {g.cues.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <Stat label="Volume" value={formatDb(p.volumeDb)} />
        <Stat label="F0" value={formatHz(p.f0)} />
        <Stat label="Confidence" value={formatPct(p.confidence)} />
        <Stat label="F1" value={formatHz(p.formants.f1)} />
        <Stat label="F2" value={formatHz(p.formants.f2)} />
        <Stat label="F3" value={formatHz(p.formants.f3)} />
      </div>

      <div>
        <p className="mb-2 text-xs text-muted">Subphonemes — start, center, end</p>
        <div className="grid gap-2">
          {p.subphonemes.map((s) => (
            <SubRow key={s.phase} sub={s} />
          ))}
        </div>
      </div>

      <div>
        <p className="mb-2 text-xs text-muted">Voice attributes</p>
        <div className="grid gap-2">
          {ATTR.map((a) => (
            <div key={a.key} title={a.hint}>
              <div className="mb-1 flex justify-between text-xs">
                <span className="text-fg">{a.label}</span>
                <span className="font-mono text-subtle">{formatPct(p.attributes[a.key])}</span>
              </div>
              <div className="h-1 rounded-full bg-surface-2">
                <div
                  className="h-full rounded-full bg-accent"
                  style={{ width: `${Math.round(p.attributes[a.key] * 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-surface-2 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wider text-subtle">{label}</div>
      <div className="mt-0.5 font-mono text-sm tabular-nums text-fg">{value}</div>
    </div>
  );
}

function SubRow({ sub }: { sub: Subphoneme }) {
  const arrow =
    sub.contour === "rise"
      ? "↗ rise"
      : sub.contour === "fall"
        ? "↘ drop"
        : sub.contour === "rise-fall"
          ? "∧ rise–drop"
          : sub.contour === "fall-rise"
            ? "∨ drop–rise"
            : "→ level";
  return (
    <div className="flex items-center gap-3 rounded-md bg-surface-2 px-3 py-2">
      <span
        className={cn(
          "w-16 shrink-0 text-xs font-medium capitalize",
          sub.phase === "nucleus" ? "text-accent" : "text-muted",
        )}
      >
        {sub.phase}
      </span>
      <span className="flex-1 font-mono text-xs text-fg">{arrow}</span>
      <span className="font-mono text-[11px] tabular-nums text-subtle">
        F0 {formatHz(sub.f0)} · {formatDb(sub.energyDb)}
      </span>
    </div>
  );
}
