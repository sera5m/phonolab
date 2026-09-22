import { clamp, lerp } from "@/lib/utils";
import {
  extractTrack,
  mean,
  median,
  slope,
  spectrogramMatrix,
  stdev,
  type FrameFeatures,
  type PitchRangeId,
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
  pitchRange?: PitchRangeId;
  transcript?: TranscriptHint;
  known?: KnownSpan[];
};

export async function analyzeVoice(input: AnalyzeInput): Promise<AnalysisResult> {
  const track = await extractTrack(input.samples, input.sampleRate, input.pitchRange ?? "speech");
  const voicedF0 = track.frames.filter((f) => f.voiced && f.f0 > 50).map((f) => f.f0);
  const meanF0 = voicedF0.length ? median(voicedF0) : 0;
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

  const overallGender = blendGender(phonemes);
  const vowelFrames = track.frames.filter((f) => f.voiced && f.f1 > 80 && f.f2 > 80);
  const meanF1 = mean(vowelFrames.map((f) => f.f1));
  const meanF2 = mean(vowelFrames.map((f) => f.f2));
  const meanF3 = mean(vowelFrames.filter((f) => f.f3 > 1500).map((f) => f.f3));
  const vtlCm = meanF3 > 1500 ? (5 * 35000) / (4 * meanF3) : 0;
  const bassRatio = measureBassRatio(track);
  const micNote = micWarning(bassRatio, meanF0);
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
    meanF1,
    meanF2,
    meanF3,
    vtlCm,
    bassRatio,
    micNote,
    pitchRange: input.pitchRange ?? "speech",
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
  const f0s = use.filter((f) => f.f0 > 50).map((f) => f.f0);
  const f1s = use.filter((f) => f.f1 > 80).map((f) => f.f1);
  const f2s = use.filter((f) => f.f2 > 80).map((f) => f.f2);
  const f3s = use.filter((f) => f.f3 > 1500).map((f) => f.f3);
  const dbs = use.map((f) => f.db);
  const formants = {
    f1: mean(f1s),
    f2: mean(f2s),
    f3: mean(f3s),
  };
  const f0 = f0s.length ? median(f0s) : 0;
  const volumeDb = mean(dbs);
  const gender = scoreGender(f0, formants, use, meta.ipa);
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
  _frames: FrameFeatures[],
  ipa: string,
): GenderScore {
  const cues: string[] = [];

  // Pitch is vocal-fold rate. 160 Hz is the middle of the overlap, not "female".
  // Typical modal speech: adult male ~85–155 Hz (mean ~120), adult female ~165–255 (mean ~210).
  let pitchCue = 0;
  let pitchW = 0;
  if (f0 > 50) {
    pitchCue = clamp((f0 - 160) / 70, -1.25, 1.25);
    pitchW = 0.7;
    if (f0 < 145) {
      cues.push(`Pitch ${Math.round(f0)} Hz — typical adult-male speaking range (~85–155 Hz)`);
    } else if (f0 > 185) {
      cues.push(`Pitch ${Math.round(f0)} Hz — typical adult-female speaking range (~165–255 Hz)`);
    } else {
      cues.push(
        `Pitch ${Math.round(f0)} Hz is in the overlap. Pitch alone cannot call this masculine or feminine.`,
      );
    }
  }

  // F1 and F2 encode the vowel (height × frontness). Using their average as
  // "gender" is why front vowels used to read feminine. F3 tracks tract length.
  const f3 = formants.f3;
  let resonanceCue = 0;
  let resW = 0;
  let vtlCm = 0;
  if (f3 > 1500) {
    vtlCm = (5 * 35000) / (4 * f3);
    resonanceCue = clamp((f3 - 2750) / 450, -1.25, 1.25);
    resW = 0.3;
    cues.push(
      vtlCm > 16.6
        ? `F3 ${Math.round(f3)} Hz ≈ ${vtlCm.toFixed(1)} cm tract — lower, longer-tube resonance`
        : vtlCm < 15.2
          ? `F3 ${Math.round(f3)} Hz ≈ ${vtlCm.toFixed(1)} cm tract — higher, shorter-tube resonance`
          : `F3 ${Math.round(f3)} Hz ≈ ${vtlCm.toFixed(1)} cm tract — mid resonance`,
    );
  }

  let vowelW = 0;
  let vowelCue = 0;
  if (formants.f1 > 80 && formants.f2 > 400) {
    const { vowel } = nearestVowel(formants.f1, formants.f2, 0.5);
    const distM = Math.hypot(formants.f1 - vowel.male[0], (formants.f2 - vowel.male[1]) * 0.55);
    const distF = Math.hypot(formants.f1 - vowel.female[0], (formants.f2 - vowel.female[1]) * 0.55);
    vowelCue = clamp((distM - distF) / 280, -1, 1);
    vowelW = 0.06;
    cues.push(
      `F1×F2 ${Math.round(formants.f1)}×${Math.round(formants.f2)} Hz on /${ipa || vowel.ipa}/ is closer to typical ${
        distM < distF ? "male" : "female"
      } targets for that vowel`,
    );
  }

  const wsum = pitchW + resW + vowelW || 1;
  const score = clamp((pitchCue * pitchW + resonanceCue * resW + vowelCue * vowelW) / wsum, -1, 1);
  const inOverlap = f0 > 145 && f0 < 185;
  const confidence = clamp(
    (pitchW ? 0.48 : 0.16) + Math.abs(score) * 0.28 - (inOverlap ? 0.24 : 0),
    0.16,
    0.9,
  );
  const label: GenderScore["label"] =
    score > 0.24 ? "feminine-coded" : score < -0.24 ? "masculine-coded" : "androgynous";
  const formantScale = vtlCm > 0 ? 16.5 / vtlCm : 1;
  return {
    score,
    label,
    confidence,
    f0Hz: f0,
    formantScale,
    cues,
    pitchCue: clamp(pitchCue, -1, 1),
    resonanceCue: clamp(resonanceCue, -1, 1),
    vtlCm,
    f3Hz: f3,
  };
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

function blendGender(phonemes: PhonemeHit[]): GenderScore {
  const vowels = phonemes.filter(
    (p) => p.f0 > 50 && (p.manner === "vowel" || p.manner === "diphthong"),
  );
  const use = vowels.length ? vowels : phonemes.filter((p) => p.f0 > 50);
  if (!use.length) {
    return {
      score: 0,
      label: "androgynous",
      confidence: 0.2,
      f0Hz: 0,
      formantScale: 1,
      cues: ["No voiced vowels to score"],
      pitchCue: 0,
      resonanceCue: 0,
      vtlCm: 0,
      f3Hz: 0,
    };
  }
  const f0Hz = median(use.map((s) => s.gender.f0Hz).filter((x) => x > 0));
  const f3Hz = mean(use.map((s) => s.gender.f3Hz).filter((x) => x > 1500));
  // Clip-level gender is pitch + F3 only. Averaging every vowel's F1×F2
  // is a mush that used to pull /i/ toward feminine.
  return scoreGender(f0Hz, { f1: 0, f2: 0, f3: f3Hz }, [], "");
}

function measureBassRatio(track: Track) {
  const binHz = track.sampleRate / track.fftSize;
  let bass = 0;
  let mid = 0;
  const b1 = Math.max(2, Math.round(120 / binHz));
  const m0 = Math.round(200 / binHz);
  const m1 = Math.round(500 / binHz);
  for (const f of track.frames) {
    if (f.db < -42) continue;
    for (let k = 1; k <= b1 && k < f.spectrum.length; k++) bass += f.spectrum[k]!;
    for (let k = m0; k <= m1 && k < f.spectrum.length; k++) mid += f.spectrum[k]!;
  }
  return mid > 0 ? bass / mid : 1;
}

function micWarning(bassRatio: number, meanF0: number) {
  if (bassRatio < 0.26 && meanF0 > 170) {
    return "Very little energy below 120 Hz, and pitch is high. A headset mic often high-passes the fundamental, so the tracker reports 2×F0 (110 Hz reads as 220). Switch the range to Chest, or this is genuinely a high speaking pitch.";
  }
  if (bassRatio < 0.2) {
    return "This clip is thin in the bass. F1 and low pitch are less reliable on a high-passed mic. Get closer, or use Chest range if the pitch looks twice what you expect.";
  }
  return null;
}

function percentile(values: number[], p: number) {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  const i = clamp(Math.floor(p * (s.length - 1)), 0, s.length - 1);
  return s[i]!;
}

export { extractTrack };
