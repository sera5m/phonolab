import type { Manner, PhonemeHit } from "@/lib/phonetics/types";
import { cn, formatPct } from "@/lib/utils";

const MANNER_TINT: Record<Manner, string> = {
  vowel: "bg-accent/20 text-accent",
  diphthong: "bg-accent/25 text-accent",
  stop: "bg-steel/25 text-fg",
  fricative: "bg-warn/15 text-warn",
  nasal: "bg-surface-2 text-muted",
  approximant: "bg-steel/15 text-muted",
  affricate: "bg-warn/20 text-warn",
  silence: "bg-bg text-subtle",
};

type Props = {
  phonemes: PhonemeHit[];
  duration: number;
  selectedId: string | null;
  onSelect: (id: string) => void;
};

export function PhonemeTimeline({ phonemes, duration, selectedId, onSelect }: Props) {
  const dur = duration || 1;
  return (
    <div className="flex flex-col gap-2">
      <div className="relative h-16 overflow-hidden rounded-md bg-surface-2 px-1">
        {phonemes.map((p) => {
          const left = (p.start / dur) * 100;
          const rawW = Math.max(3.2, ((p.end - p.start) / dur) * 100);
          const width = Math.min(rawW, Math.max(2, 99.2 - left));
          const selected = p.id === selectedId;
          return (
            <button
              key={p.id}
              type="button"
              onClick={() => onSelect(p.id)}
              className={cn(
                "absolute top-2 flex h-12 flex-col items-center justify-center overflow-hidden rounded-sm px-0.5 font-display text-lg leading-none transition-[transform,box-shadow] duration-150",
                MANNER_TINT[p.manner],
                selected && "z-10 ring-1 ring-accent",
              )}
              style={{ left: `${left}%`, width: `${width}%` }}
              aria-pressed={selected}
              title={`/${p.ipa}/ ${formatPct(p.confidence)}`}
            >
              <span className="truncate">{p.ipa === " " ? "∅" : p.ipa}</span>
              <span className="mt-0.5 h-0.5 w-8/12 rounded-full bg-fg/20">
                <span
                  className="block h-full rounded-full bg-fg/70"
                  style={{ width: `${Math.round(p.confidence * 100)}%` }}
                />
              </span>
            </button>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-subtle">
        <span className="inline-flex items-center gap-1.5">
          <i className="size-2 rounded-sm bg-accent/50" /> vowel
        </span>
        <span className="inline-flex items-center gap-1.5">
          <i className="size-2 rounded-sm bg-warn/50" /> fricative
        </span>
        <span className="inline-flex items-center gap-1.5">
          <i className="size-2 rounded-sm bg-steel/50" /> stop
        </span>
        <span className="inline-flex items-center gap-1.5">
          <i className="size-2 rounded-sm bg-muted/40" /> nasal / glide
        </span>
      </div>
    </div>
  );
}
