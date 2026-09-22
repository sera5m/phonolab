import type { AnalysisResult, PhonemeHit } from "@/lib/phonetics/types";
import { VOWEL_TARGETS } from "@/lib/phonetics/data";
import { cn } from "@/lib/utils";

type Props = {
  analysis: AnalysisResult;
  selected: PhonemeHit | null;
  onSelect: (id: string) => void;
};

const F2_MAX = 3100;
const F2_MIN = 600;
const F1_MIN = 200;
const F1_MAX = 1000;

function xOf(f2: number) {
  return ((F2_MAX - f2) / (F2_MAX - F2_MIN)) * 100;
}
function yOf(f1: number) {
  return ((f1 - F1_MIN) / (F1_MAX - F1_MIN)) * 100;
}

export function FormantSpace({ analysis, selected, onSelect }: Props) {
  const vowels = analysis.phonemes.filter(
    (p) => (p.manner === "vowel" || p.manner === "diphthong") && p.formants.f1 > 80 && p.formants.f2 > 400,
  );

  return (
    <div>
      <div className="mb-2 flex items-end justify-between gap-3">
        <div>
          <h2 className="text-sm font-medium text-fg">F1 × F2 — vowel space</h2>
          <p className="mt-0.5 text-xs text-muted">
            The frequency pair. F1 is tongue height (open is down). F2 is front/back (front is left).
          </p>
        </div>
        <div className="flex gap-3 font-mono text-[11px] text-subtle">
          <span className="inline-flex items-center gap-1.5">
            <span className="size-1.5 rounded-full bg-steel" />
            male
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="size-1.5 rounded-full bg-accent" />
            female
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="size-1.5 rounded-full bg-fg" />
            this clip
          </span>
        </div>
      </div>
      <div className="relative h-64 w-full rounded-md bg-surface-2 sm:h-72">
        <svg viewBox="0 0 100 100" className="size-full" role="img" aria-label="F1 F2 vowel chart">
          <text x="50" y="6" textAnchor="middle" className="fill-subtle" fontSize="3.2">
            front ← F2 → back
          </text>
          <text
            x="4"
            y="52"
            textAnchor="middle"
            className="fill-subtle"
            fontSize="3.2"
            transform="rotate(-90 4 52)"
          >
            close ← F1 → open
          </text>
          {VOWEL_TARGETS.map((v) => (
            <g key={v.ipa}>
              <line
                x1={xOf(v.male[1])}
                y1={yOf(v.male[0])}
                x2={xOf(v.female[1])}
                y2={yOf(v.female[0])}
                className="stroke-border"
                strokeWidth="0.35"
              />
              <circle cx={xOf(v.male[1])} cy={yOf(v.male[0])} r="1.1" className="fill-steel" />
              <circle cx={xOf(v.female[1])} cy={yOf(v.female[0])} r="1.1" className="fill-accent" />
              <text
                x={xOf((v.male[1] + v.female[1]) / 2)}
                y={yOf((v.male[0] + v.female[0]) / 2) - 1.8}
                textAnchor="middle"
                className="fill-muted"
                fontSize="3.4"
                fontFamily="Georgia, serif"
              >
                {v.ipa}
              </text>
            </g>
          ))}
          {vowels.map((p) => {
            const active = selected?.id === p.id;
            return (
              <g
                key={p.id}
                className="cursor-pointer"
                onClick={() => onSelect(p.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") onSelect(p.id);
                }}
                role="button"
                tabIndex={0}
              >
                <circle
                  cx={xOf(p.formants.f2)}
                  cy={yOf(p.formants.f1)}
                  r={active ? 2.6 : 2}
                  className={cn(active ? "fill-fg" : "fill-warn")}
                />
                {active ? (
                  <text
                    x={xOf(p.formants.f2)}
                    y={yOf(p.formants.f1) + 6}
                    textAnchor="middle"
                    className="fill-fg"
                    fontSize="3.6"
                    fontFamily="ui-monospace, monospace"
                  >
                    {Math.round(p.formants.f1)}×{Math.round(p.formants.f2)}
                  </text>
                ) : null}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
