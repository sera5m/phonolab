import { clamp, lerp } from "@/lib/utils";
import {
  extractTrack,
  mean,
  slope,
  spectrogramMatrix,
  stdev,
  type FrameFeatures,
  type Track,
} from "@/lib/audio/dsp";
import type { SynthPhoneme } from "@/lib/audio/synth";
import { nearestVowel } from "./data";
import { textToPhones, wordToPhones } from "./g2p";
import type {
  AnalysisResult,
  AttributeScores,
  Contour,
  GenderScore,
  Manner,
  PhonemeHit,
  Subphoneme,
  TranscriptHint,
  WordHit,
} from "./types";

export type KnownSpan = {
  ipa: string;
  start: number;
  end: number;
  spec?: SynthPhoneme;
};

export type AnalyzeInput = {
  samples: Float32Array;
  sampleRate: number;
  sourceName: string;
  backend: "webgpu" | "cpu";
  transcript?: TranscriptHint;
  known?: KnownSpan[];
};

export async function analyzeVoice(input: AnalyzeInput): Promise<AnalysisResult> {
  const track = await extractTrack(input.samples, input.sampleRate);
  const voicedF0 = track.frames.filter((f) => f.voiced && f.f0 > 60).map((f) => f.f0);
  const meanF0 = voicedF0.length ? mean(voicedF0) : 0;
  const meanVolumeDb = mean(track.frames.map((f) => f.db));

  let phonemes: PhonemeHit[];
  let words: WordHit[] = input.transcript?.words ?? [];
  let transcript = input.transcript?.text ?? "";

  if (input.known?.length) {
    phonemes = input.known.map((span, i) =>
      scoreSpan(track, span.start, span.end, {
        id: `p-${i}`,
        ipa: span.ipa,
        arpabet: span.spec?.arpabet ?? span.ipa.toUpperCase(),
        example: span.spec?.example ?? "",
        manner: span.spec?.manner ?? inferManner(span.ipa),
        word: span.spec?.example,
        prior: 0.92,
      }),
    );
    if (!transcript) transcript = input.known.map((k) => k.ipa).join(" ");
  } else if (input.transcript?.text) {
    phonemes = alignTranscript(track, input.transcript);
    transcript = input.transcript.text;
    if (!words.length) words = inferWordSpans(track, input.transcript.text, phonemes);
  } else {
    phonemes = acousticSegments(track, meanF0);
    transcript = phonemes.map((p) => p.ipa).join(" ");
  }

  phonemes = phonemes.filter((p) => p.end - p.start > 0.018 && p.manner !== "silence");

  const overallGender = blendGender(phonemes.map((p) => p.gender));
  const spec = spectrogramMatrix(track);
  const waveN = Math.min(1200, input.samples.length);
  const waveform = new Float32Array(waveN);
  const step = input.samples.length / waveN;
  for (let i = 0; i < waveN; i++) {
    const i0 = Math.floor(i * step);
    const i1 = Math.min(input.samples.length, Math.floor((i + 1) * step));
    let peak = 0;
    for (let j = i0; j < i1; j++) peak = Math.max(peak, Math.abs(input.samples[j] ?? 0));
    waveform[i] = peak;
  }
  const formantTrack = track.frames
    .filter((_, i) => i % 2 === 0)
    .map((f) => ({ t: f.t, f1: f.f1, f2: f.f2, f3: f.f3, f0: f.f0 }));

  return {
    duration: track.duration,
    sampleRate: track.sampleRate,
    transcript,
    words,
    phonemes,
    meanF0,
    meanVolumeDb,
    overallGender,
    backend: input.backend,
    sourceName: input.sourceName,
    spectrogram: spec,
    waveform,
    formantTrack,
  };
}

function inferManner(ipa: string): Manner {
  if ("iɪeɛæɑɔoʊuʌəɝyø".includes(ipa[0] ?? "")) return "vowel";
  if (["eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ"].includes(ipa)) return "diphthong";
  if ("ptkbdg".includes(ipa)) return "stop";
  if ("fvθðszʃʒh".includes(ipa)) return "fricative";
  if ("mnŋ".includes(ipa)) return "nasal";
  if ("tʃdʒ".includes(ipa)) return "affricate";
  return "approximant";
}

function framesIn(track: Track, start: number, end: number) {
  return track.frames.filter((f) => f.t >= start && f.t < end);
}

function scoreSpan(
  track: Track,
  start: number,
  end: number,
  meta: {
    id: string;
    ipa: string;
    arpabet: string;
    example: string;
    manner: Manner;
    word?: string;
    prior: number;
  },
): PhonemeHit {
  const frames = framesIn(track, start, end);
  const use = frames.length ? frames : nearestFrame(track, (start + end) / 2);
  const f0s = use.filter((f) => f.f0 > 60).map((f) => f.f0);
  const f1s = use.filter((f) => f.f1 > 80).map((f) => f.f1);
  const f2s = use.filter((f) => f.f2 > 80).map((f) => f.f2);
  const f3s = use.filter((f) => f.f3 > 80).map((f) => f.f3);
  const dbs = use.map((f) => f.db);
  const formants = {
    f1: mean(f1s),
    f2: mean(f2s),
    f3: mean(f3s),
  };
  const f0 = mean(f0s);
  const volumeDb = mean(dbs);
  const gender = scoreGender(f0, formants, use);
  const attributes = scoreAttributes(use, f0);
  const subphonemes = splitSubphonemes(use, start, end);
  const identConf =
    meta.manner === "vowel" || meta.manner === "diphthong"
      ? nearestVowel(formants.f1 || 500, formants.f2 || 1500, (gender.score + 1) / 2).confidence
      : clamp(0.45 + (1 - stdev(use.map((f) => f.centroid)) / 4000), 0.35, 0.9);
  const confidence = clamp(meta.prior * 0.55 + identConf * 0.45, 0.2, 0.98);

  return {
    id: meta.id,
    ipa: meta.ipa,
    arpabet: meta.arpabet,
    example: meta.example,
    manner: meta.manner,
    start,
    end,
    confidence,
    volumeDb,
    gender,
    attributes,
    subphonemes,
    formants,
    f0,
    word: meta.word,
  };
}

function nearestFrame(track: Track, t: number): FrameFeatures[] {
  if (!track.frames.length) return [];
  let best = track.frames[0]!;
  let d = Infinity;
  for (const f of track.frames) {
    const dd = Math.abs(f.t - t);
    if (dd < d) {
      d = dd;
      best = f;
    }
  }
  return [best];
}

function scoreGender(
  f0: number,
  formants: { f1: number; f2: number; f3: number },
  frames: FrameFeatures[],
): GenderScore {
  const cues: string[] = [];
  let f0Score = 0;
  let f0W = 0;
  if (f0 > 60) {
    f0Score = clamp((f0 - 145) / 80, -1.2, 1.2);
    f0W = 0.58;
    if (f0 < 150) cues.push(`F0 ${Math.round(f0)} Hz sits in the typical adult-male range`);
    else if (f0 > 185) cues.push(`F0 ${Math.round(f0)} Hz sits in the typical adult-female range`);
    else cues.push(`F0 ${Math.round(f0)} Hz is in the overlap zone — weaker cue`);
  }
  const usable = [formants.f1, formants.f2, formants.f3].filter((x) => x > 80);
  const meanF = usable.length ? mean(usable) : 0;
  let formantScore = 0;
  let formantW = 0;
  let formantScale = 1;
  if (meanF > 0) {
    formantScale = meanF / 1500;
    formantScore = clamp((meanF - 1550) / 420, -1.2, 1.2);
    formantW = 0.32;
    cues.push(
      formantScale > 1.08
        ? "Higher formants — shorter vocal-tract resonance"
        : formantScale < 0.94
          ? "Lower formants — longer vocal-tract resonance"
          : "Formant scale near the mid talker average",
    );
  }
  const tiltFrames = frames.filter((f) => f.spectrum.length > 20);
  let tiltScore = 0;
  let tiltW = 0;
  if (tiltFrames.length) {
    const tilts = tiltFrames.map((f) => spectralTilt(f));
    const tilt = mean(tilts);
    tiltScore = clamp((-tilt - 6) / 10, -1, 1);
    tiltW = 0.1;
  }
  const wsum = f0W + formantW + tiltW || 1;
  const score = clamp((f0Score * f0W + formantScore * formantW + tiltScore * tiltW) / wsum, -1, 1);
  const inOverlap = f0 > 145 && f0 < 185;
  const confidence = clamp((f0W ? 0.55 : 0.2) + Math.abs(score) * 0.35 - (inOverlap ? 0.18 : 0), 0.22, 0.95);
  const label: GenderScore["label"] =
    score > 0.18 ? "feminine-coded" : score < -0.18 ? "masculine-coded" : "androgynous";
  return { score, label, confidence, f0Hz: f0, formantScale, cues };
}

function spectralTilt(frame: FrameFeatures) {
  const n = frame.spectrum.length;
  const lowN = Math.max(2, Math.floor(n * 0.12));
  const highN = Math.max(2, Math.floor(n * 0.45));
  let low = 0;
  let high = 0;
  for (let i = 1; i < lowN; i++) low += frame.spectrum[i]!;
  for (let i = n - highN; i < n; i++) high += frame.spectrum[i]!;
  low /= lowN;
  high /= highN;
  return 20 * Math.log10((high + 1e-8) / (low + 1e-8));
}

function scoreAttributes(frames: FrameFeatures[], f0: number): AttributeScores {
  if (!frames.length) {
    return { brightness: 0, breathiness: 0, nasality: 0, roughness: 0, tension: 0, resonance: 0 };
  }
  const centroid = mean(frames.map((f) => f.centroid));
  const hnr = mean(frames.map((f) => f.hnr));
  const zcr = mean(frames.map((f) => f.zcr));
  const f1 = mean(frames.filter((f) => f.f1 > 80).map((f) => f.f1));
  const f0s = frames.filter((f) => f.f0 > 60).map((f) => f.f0);
  const rms = frames.map((f) => f.rms);
  const jitter = f0s.length > 2 ? stdev(f0s) / (mean(f0s) || 1) : 0;
  const shimmer = rms.length > 2 ? stdev(rms) / (mean(rms) || 1e-6) : 0;

  const brightness = clamp(centroid / 3500, 0, 1);
  const breathiness = clamp((12 - hnr) / 24 + zcr * 1.4, 0, 1);
  const nasality = clamp(f1 > 0 ? (1 - Math.abs(f1 - 280) / 500) * 0.5 + (f1 < 450 ? 0.2 : 0) : 0, 0, 1);
  const roughness = clamp(jitter * 8 + shimmer * 3, 0, 1);
  const tension = clamp((f0 > 0 ? (f0 - 120) / 220 : 0) + (f1 > 0 ? (700 - f1) / 900 : 0), 0, 1);
  const peakiness = mean(
    frames.map((f) => {
      let mx = 0;
      let sum = 0;
      for (let i = 0; i < f.spectrum.length; i++) {
        mx = Math.max(mx, f.spectrum[i]!);
        sum += f.spectrum[i]!;
      }
      return sum > 0 ? mx / (sum / f.spectrum.length) : 0;
    }),
  );
  const resonance = clamp((peakiness - 1.5) / 8, 0, 1);

  return { brightness, breathiness, nasality, roughness, tension, resonance };
}

function contourFrom(start: number, mid: number, end: number, scale: number): Contour {
  const up = scale * 0.08;
  const riseStart = mid - start > up;
  const fallStart = start - mid > up;
  const riseEnd = end - mid > up;
  const fallEnd = mid - end > up;
  if (riseStart && fallEnd) return "rise-fall";
  if (fallStart && riseEnd) return "fall-rise";
  if (end - start > up) return "rise";
  if (start - end > up) return "fall";
  return "level";
}

function splitSubphonemes(frames: FrameFeatures[], start: number, end: number): Subphoneme[] {
  const dur = Math.max(end - start, 0.001);
  const cuts: [Subphoneme["phase"], number, number][] = [
    ["onset", start, start + dur * 0.28],
    ["nucleus", start + dur * 0.28, start + dur * 0.72],
    ["coda", start + dur * 0.72, end],
  ];
  return cuts.map(([phase, a, b]) => {
    const slice = frames.filter((f) => f.t >= a && f.t < b);
    const use = slice.length ? slice : frames;
    const f0s = use.map((f) => (f.f0 > 60 ? f.f0 : 0));
    const dbs = use.map((f) => f.db);
    const f1 = mean(use.filter((f) => f.f1 > 80).map((f) => f.f1));
    const f2 = mean(use.filter((f) => f.f2 > 80).map((f) => f.f2));
    const f3 = mean(use.filter((f) => f.f3 > 80).map((f) => f.f3));
    const f0 = mean(f0s.filter((x) => x > 0));
    const energyDb = mean(dbs);
    const f0Slope = slope(f0s.filter((x) => x > 0).length > 1 ? f0s.filter((x) => x > 0) : f0s);
    const energySlope = slope(dbs);
    const thirds = splitThird(use);
    const contour = contourFrom(thirds[0], thirds[1], thirds[2], Math.max(f0, 80));
    return {
      phase,
      start: a,
      end: b,
      f0,
      f0Slope,
      f1,
      f2,
      f3,
      energyDb,
      energySlope,
      contour,
    };
  });
}

function splitThird(frames: FrameFeatures[]) {
  if (!frames.length) return [0, 0, 0] as const;
  const vals = frames.map((f) => (f.f0 > 60 ? f.f0 : f.db + 80));
  const a = vals.slice(0, Math.max(1, Math.floor(vals.length / 3)));
  const b = vals.slice(Math.floor(vals.length / 3), Math.max(1, Math.floor((2 * vals.length) / 3)));
  const c = vals.slice(Math.floor((2 * vals.length) / 3));
  return [mean(a), mean(b.length ? b : a), mean(c.length ? c : a)] as const;
}

function acousticSegments(track: Track, meanF0: number): PhonemeHit[] {
  const frames = track.frames;
  if (!frames.length) return [];
  const dbs = frames.map((f) => f.db);
  const noiseFloor = percentile(dbs, 0.2);
  const fluxThresh = percentile(frames.map((f) => f.flux), 0.78);
  const bounds: number[] = [0];
  for (let i = 2; i < frames.length - 1; i++) {
    const f = frames[i]!;
    const prev = frames[i - 1]!;
    const voicedFlip = f.voiced !== prev.voiced;
    const loudFlip = f.db > noiseFloor + 8 !== prev.db > noiseFloor + 8;
    const fluxPeak = f.flux > fluxThresh && f.flux > prev.flux && f.flux >= (frames[i + 1]?.flux ?? 0);
    if ((voicedFlip || loudFlip || fluxPeak) && f.t - frames[bounds[bounds.length - 1]!]!.t > 0.035) {
      bounds.push(i);
    }
  }
  bounds.push(frames.length);

  const feminineBias = meanF0 > 0 ? clamp((meanF0 - 145) / 80, 0, 1) : 0.5;
  const hits: PhonemeHit[] = [];
  for (let b = 0; b < bounds.length - 1; b++) {
    const i0 = bounds[b]!;
    const i1 = bounds[b + 1]!;
    const slice = frames.slice(i0, i1);
    if (!slice.length) continue;
    const start = slice[0]!.t;
    const end = (frames[i1]?.t ?? start + 0.04);
    if (mean(slice.map((f) => f.db)) < noiseFloor + 6) continue;
    const label = classifySlice(slice, feminineBias);
    hits.push(
      scoreSpan(track, start, end, {
        id: `p-${b}`,
        ipa: label.ipa,
        arpabet: label.arpabet,
        example: label.example,
        manner: label.manner,
        prior: label.confidence,
      }),
    );
  }
  return hits;
}

function classifySlice(slice: FrameFeatures[], feminineBias: number) {
  const zcr = mean(slice.map((f) => f.zcr));
  const centroid = mean(slice.map((f) => f.centroid));
  const voiced = slice.filter((f) => f.voiced).length / slice.length;
  const f1 = mean(slice.filter((f) => f.f1 > 80).map((f) => f.f1));
  const f2 = mean(slice.filter((f) => f.f2 > 80).map((f) => f.f2));
  const db = mean(slice.map((f) => f.db));
  const dur = (slice[slice.length - 1]!.t - slice[0]!.t) || 0.04;

  if (zcr > 0.22 && centroid > 2800) {
    const ipa = centroid > 4500 ? "s" : centroid > 3500 ? "ʃ" : "f";
    const arpa = ipa === "s" ? "S" : ipa === "ʃ" ? "SH" : "F";
    return {
      ipa,
      arpabet: arpa,
      example: ipa === "s" ? "see" : ipa === "ʃ" ? "she" : "fee",
      manner: "fricative" as const,
      confidence: 0.72,
    };
  }
  if (voiced < 0.25 && dur < 0.09 && db < -22) {
    return { ipa: "t", arpabet: "T", example: "tie", manner: "stop" as const, confidence: 0.55 };
  }
  if (voiced > 0.35 && f1 > 150 && f1 < 500 && f2 < 1400 && zcr < 0.12) {
    return { ipa: "m", arpabet: "N", example: "me", manner: "nasal" as const, confidence: 0.6 };
  }
  if (voiced > 0.4 && f1 > 80 && f2 > 400) {
    const { vowel, confidence } = nearestVowel(f1, f2, feminineBias);
    return {
      ipa: vowel.ipa,
      arpabet: vowel.arpabet,
      example: vowel.example,
      manner: "vowel" as const,
      confidence,
    };
  }
  if (voiced > 0.3) {
    return { ipa: "ə", arpabet: "AH", example: "comma", manner: "vowel" as const, confidence: 0.4 };
  }
  return { ipa: "h", arpabet: "HH", example: "he", manner: "fricative" as const, confidence: 0.35 };
}

function alignTranscript(track: Track, hint: TranscriptHint): PhonemeHit[] {
  const duration = track.duration;
  if (hint.words?.length) {
    const hits: PhonemeHit[] = [];
    let n = 0;
    for (const w of hint.words) {
      const phones = wordToPhones(w.word);
      const weights = phones.map((p) => (p.manner === "vowel" || p.manner === "diphthong" ? 1.45 : 0.8));
      const sum = weights.reduce((a, b) => a + b, 0) || 1;
      let t = w.start;
      phones.forEach((p, i) => {
        const dur = ((w.end - w.start) * weights[i]!) / sum;
        hits.push(
          scoreSpan(track, t, t + dur, {
            id: `p-${n++}`,
            ipa: p.ipa,
            arpabet: p.arpabet,
            example: p.example,
            manner: p.manner,
            word: p.word,
            prior: 0.78,
          }),
        );
        t += dur;
      });
    }
    return hits;
  }
  const phones = textToPhones(hint.text);
  const speech = speechBounds(track);
  const start = speech.start;
  const end = speech.end;
  const weights = phones.map((p) => (p.manner === "vowel" || p.manner === "diphthong" ? 1.45 : 0.8));
  const sum = weights.reduce((a, b) => a + b, 0) || 1;
  let t = start;
  return phones.map((p, i) => {
    const dur = ((end - start) * weights[i]!) / sum;
    const hit = scoreSpan(track, t, t + dur, {
      id: `p-${i}`,
      ipa: p.ipa,
      arpabet: p.arpabet,
      example: p.example,
      manner: p.manner,
      word: p.word,
      prior: 0.7,
    });
    t += dur;
    return hit;
  });
}

function speechBounds(track: Track) {
  const dbs = track.frames.map((f) => f.db);
  const thr = percentile(dbs, 0.3) + 8;
  let i0 = 0;
  let i1 = track.frames.length - 1;
  while (i0 < i1 && (track.frames[i0]?.db ?? -99) < thr) i0++;
  while (i1 > i0 && (track.frames[i1]?.db ?? -99) < thr) i1--;
  return {
    start: track.frames[i0]?.t ?? 0,
    end: track.frames[i1]?.t ?? track.duration,
  };
}

function inferWordSpans(track: Track, text: string, phonemes: PhonemeHit[]): WordHit[] {
  const tokens = text.split(/\s+/).filter(Boolean);
  if (!tokens.length || !phonemes.length) return [];
  const start = phonemes[0]!.start;
  const end = phonemes[phonemes.length - 1]!.end;
  return tokens.map((word, i) => ({
    word,
    start: lerp(start, end, i / tokens.length),
    end: lerp(start, end, (i + 1) / tokens.length),
  }));
}

function blendGender(scores: GenderScore[]): GenderScore {
  if (!scores.length) {
    return {
      score: 0,
      label: "androgynous",
      confidence: 0.2,
      f0Hz: 0,
      formantScale: 1,
      cues: ["No voiced frames"],
    };
  }
  const score = mean(scores.map((s) => s.score));
  const confidence = mean(scores.map((s) => s.confidence));
  const f0Hz = mean(scores.map((s) => s.f0Hz).filter((x) => x > 0));
  const formantScale = mean(scores.map((s) => s.formantScale));
  const label: GenderScore["label"] =
    score > 0.18 ? "feminine-coded" : score < -0.18 ? "masculine-coded" : "androgynous";
  const cues = [
    `Mean F0 ${f0Hz ? Math.round(f0Hz) + " Hz" : "unvoiced"}`,
    `Formant scale ${formantScale.toFixed(2)}×`,
  ];
  return { score, label, confidence, f0Hz, formantScale, cues };
}

function percentile(values: number[], p: number) {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  const i = clamp(Math.floor(p * (s.length - 1)), 0, s.length - 1);
  return s[i]!;
}

export { extractTrack };
