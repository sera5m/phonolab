import { lerp } from "@/lib/utils";
import { TARGET_SR } from "./dsp";

export type SynthPhoneme = {
  ipa: string;
  arpabet: string;
  example: string;
  manner:
    | "vowel"
    | "diphthong"
    | "stop"
    | "fricative"
    | "nasal"
    | "approximant"
    | "affricate"
    | "silence";
  dur: number;
  f0: number;
  f1: number | [number, number];
  f2: number | [number, number];
  f3: number | [number, number];
  voiced: boolean;
  amp: number;
};

export type DemoClip = {
  id: string;
  name: string;
  blurb: string;
  transcript: string;
  phonemes: SynthPhoneme[];
};

function P(
  partial: Omit<SynthPhoneme, "arpabet" | "example"> & { arpabet?: string; example?: string },
): SynthPhoneme {
  return {
    arpabet: partial.arpabet ?? partial.ipa.toUpperCase(),
    example: partial.example ?? "",
    ...partial,
  };
}

const FEM = 210;
const MAL = 118;

function vowels(f0: number): SynthPhoneme[] {
  const gap = P({
    ipa: " ",
    manner: "silence",
    dur: 0.05,
    f0: 0,
    f1: 0,
    f2: 0,
    f3: 0,
    voiced: false,
    amp: 0,
  });
  const v = (
    ipa: string,
    arpabet: string,
    example: string,
    f1: number,
    f2: number,
    f3: number,
  ) =>
    P({
      ipa,
      arpabet,
      example,
      manner: "vowel",
      dur: 0.22,
      f0,
      f1,
      f2,
      f3,
      voiced: true,
      amp: 0.22,
    });
  return [
    v("i", "IY", "fleece", f0 > 160 ? 310 : 270, f0 > 160 ? 2790 : 2290, f0 > 160 ? 3310 : 3010),
    gap,
    v("ɪ", "IH", "kit", f0 > 160 ? 430 : 390, f0 > 160 ? 2480 : 1990, f0 > 160 ? 3070 : 2550),
    gap,
    v("ɛ", "EH", "dress", f0 > 160 ? 610 : 530, f0 > 160 ? 2330 : 1840, f0 > 160 ? 2990 : 2480),
    gap,
    v("æ", "AE", "trap", f0 > 160 ? 860 : 660, f0 > 160 ? 2050 : 1720, f0 > 160 ? 2850 : 2410),
    gap,
    v("ɑ", "AA", "lot", f0 > 160 ? 850 : 730, f0 > 160 ? 1220 : 1090, f0 > 160 ? 2810 : 2440),
    gap,
    v("ɔ", "AO", "thought", f0 > 160 ? 590 : 570, f0 > 160 ? 920 : 840, f0 > 160 ? 2710 : 2410),
    gap,
    v("u", "UW", "goose", f0 > 160 ? 370 : 300, f0 > 160 ? 950 : 870, f0 > 160 ? 2670 : 2240),
    gap,
    v("ʌ", "AH", "strut", f0 > 160 ? 760 : 640, f0 > 160 ? 1400 : 1190, f0 > 160 ? 2780 : 2390),
  ];
}

function phrase(f0: number): SynthPhoneme[] {
  const hi = f0 > 160;
  const f = (lo: number, hiV: number) => (hi ? hiV : lo);
  const sil = (d = 0.04) =>
    P({
      ipa: " ",
      manner: "silence",
      dur: d,
      f0: 0,
      f1: 0,
      f2: 0,
      f3: 0,
      voiced: false,
      amp: 0,
    });
  return [
    P({
      ipa: "s",
      arpabet: "S",
      example: "see",
      manner: "fricative",
      dur: 0.11,
      f0,
      f1: 0,
      f2: 4000,
      f3: 5500,
      voiced: false,
      amp: 0.09,
    }),
    P({
      ipa: "i",
      arpabet: "IY",
      example: "see",
      manner: "vowel",
      dur: 0.16,
      f0,
      f1: f(270, 310),
      f2: f(2290, 2790),
      f3: f(3010, 3310),
      voiced: true,
      amp: 0.2,
    }),
    sil(),
    P({
      ipa: "ð",
      arpabet: "DH",
      example: "the",
      manner: "fricative",
      dur: 0.07,
      f0,
      f1: 300,
      f2: 1400,
      f3: 2400,
      voiced: true,
      amp: 0.08,
    }),
    P({
      ipa: "ə",
      arpabet: "AH",
      example: "the",
      manner: "vowel",
      dur: 0.08,
      f0: f0 * 0.95,
      f1: f(500, 550),
      f2: f(1500, 1700),
      f3: f(2500, 2700),
      voiced: true,
      amp: 0.16,
    }),
    sil(0.03),
    P({
      ipa: "k",
      arpabet: "K",
      example: "cat",
      manner: "stop",
      dur: 0.06,
      f0,
      f1: 0,
      f2: 1800,
      f3: 2800,
      voiced: false,
      amp: 0.14,
    }),
    P({
      ipa: "æ",
      arpabet: "AE",
      example: "cat",
      manner: "vowel",
      dur: 0.16,
      f0: f0 * 1.04,
      f1: f(660, 860),
      f2: f(1720, 2050),
      f3: f(2410, 2850),
      voiced: true,
      amp: 0.22,
    }),
    P({
      ipa: "t",
      arpabet: "T",
      example: "cat",
      manner: "stop",
      dur: 0.07,
      f0,
      f1: 0,
      f2: 1800,
      f3: 2700,
      voiced: false,
      amp: 0.12,
    }),
    sil(0.08),
    P({
      ipa: "s",
      arpabet: "S",
      example: "sit",
      manner: "fricative",
      dur: 0.1,
      f0,
      f1: 0,
      f2: 4200,
      f3: 5600,
      voiced: false,
      amp: 0.09,
    }),
    P({
      ipa: "ɪ",
      arpabet: "IH",
      example: "sit",
      manner: "vowel",
      dur: 0.12,
      f0: f0 * 0.92,
      f1: f(390, 430),
      f2: f(1990, 2480),
      f3: f(2550, 3070),
      voiced: true,
      amp: 0.18,
    }),
    P({
      ipa: "t",
      arpabet: "T",
      example: "sit",
      manner: "stop",
      dur: 0.07,
      f0,
      f1: 0,
      f2: 1800,
      f3: 2700,
      voiced: false,
      amp: 0.11,
    }),
  ];
}

export const DEMO_CLIPS: DemoClip[] = [
  {
    id: "vowels-f",
    name: "Cardinal vowels · higher F0",
    blurb: "Eight English vowels at ~210 Hz. Formants sit higher — a shorter vocal tract.",
    transcript: "i ɪ ɛ æ ɑ ɔ u ʌ",
    phonemes: vowels(FEM),
  },
  {
    id: "vowels-m",
    name: "Cardinal vowels · lower F0",
    blurb: "The same eight vowels at ~118 Hz. Formants sit lower — a longer vocal tract.",
    transcript: "i ɪ ɛ æ ɑ ɔ u ʌ",
    phonemes: vowels(MAL),
  },
  {
    id: "see-the-cat-f",
    name: "See the cat sit · higher F0",
    blurb: "Mixed stops, fricatives, and vowels so onset / nucleus / coda actually move.",
    transcript: "see the cat sit",
    phonemes: phrase(FEM),
  },
  {
    id: "see-the-cat-m",
    name: "See the cat sit · lower F0",
    blurb: "Same phrase, lower pitch and formants. Compare gender scores phoneme by phoneme.",
    transcript: "see the cat sit",
    phonemes: phrase(MAL),
  },
];

function biquadResonator(f: number, bw: number, sr: number) {
  const r = Math.exp((-Math.PI * bw) / sr);
  const omega = (2 * Math.PI * f) / sr;
  const a1 = -2 * r * Math.cos(omega);
  const a2 = r * r;
  let y1 = 0;
  let y2 = 0;
  return (x: number) => {
    const y = x - a1 * y1 - a2 * y2;
    y2 = y1;
    y1 = y;
    return y;
  };
}

function readPair(v: number | [number, number], t: number) {
  return Array.isArray(v) ? lerp(v[0], v[1], t) : v;
}

export function synthesize(phonemes: SynthPhoneme[], sr = TARGET_SR) {
  const total = phonemes.reduce((s, p) => s + p.dur, 0) + 0.08;
  const n = Math.ceil(total * sr);
  const samples = new Float32Array(n);
  let cursor = Math.round(0.04 * sr);
  const timeline: { ipa: string; start: number; end: number; spec: SynthPhoneme }[] = [];
  let phase = 0;

  for (const p of phonemes) {
    const len = Math.max(1, Math.round(p.dur * sr));
    const start = cursor / sr;
    if (p.manner === "silence" || p.amp <= 0) {
      cursor += len;
      continue;
    }
    const r1 = biquadResonator(readPair(p.f1, 0) || 500, 90, sr);
    const r2 = biquadResonator(readPair(p.f2, 0) || 1500, 120, sr);
    const r3 = biquadResonator(readPair(p.f3, 0) || 2500, 160, sr);
    for (let i = 0; i < len; i++) {
      const t = i / (len - 1 || 1);
      const env = Math.sin(Math.PI * Math.min(1, Math.max(0, t < 0.12 ? t / 0.12 : t > 0.85 ? (1 - t) / 0.15 : 1)));
      const f0 = p.voiced ? p.f0 * (1 + 0.03 * Math.sin(2 * Math.PI * t)) : 0;
      let src = 0;
      if (p.voiced && f0 > 0) {
        phase += f0 / sr;
        if (phase >= 1) phase -= 1;
        const saw = 2 * phase - 1;
        src = saw * 0.35;
        if (phase < 0.12) src += 0.8;
      }
      if (!p.voiced || p.manner === "fricative") {
        src += (Math.random() * 2 - 1) * (p.voiced ? 0.08 : 0.55);
      }
      if (p.manner === "stop" && t < 0.18) {
        src += (Math.random() * 2 - 1) * 1.4 * (1 - t / 0.18);
      }
      let y = src;
      if (readPair(p.f1, t) > 80) y = r1(y);
      if (readPair(p.f2, t) > 80) y = r2(y);
      if (readPair(p.f3, t) > 80) y = r3(y);
      if (p.manner === "fricative") {
        y = y - (samples[cursor + i - 1] ?? 0) * 0.4;
      }
      const idx = cursor + i;
      if (idx < n) samples[idx] = Math.max(-0.95, Math.min(0.95, y * p.amp * (0.35 + 0.65 * env)));
    }
    const end = (cursor + len) / sr;
    timeline.push({ ipa: p.ipa, start, end, spec: p });
    cursor += len;
  }

  let peak = 1e-6;
  for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(samples[i]!));
  const gain = 0.7 / peak;
  for (let i = 0; i < n; i++) samples[i]! *= gain;

  return { samples, sampleRate: sr, timeline };
}
