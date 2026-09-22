export type Manner =
  | "vowel"
  | "diphthong"
  | "stop"
  | "fricative"
  | "nasal"
  | "approximant"
  | "affricate"
  | "silence";

export type Contour = "rise" | "fall" | "level" | "rise-fall" | "fall-rise";

export type SubPhase = "onset" | "nucleus" | "coda";

export type GenderLabel = "masculine-coded" | "androgynous" | "feminine-coded";

export type Subphoneme = {
  phase: SubPhase;
  start: number;
  end: number;
  f0: number;
  f1: number;
  f2: number;
  f3: number;
  f0Slope: number;
  energyDb: number;
  energySlope: number;
  contour: Contour;
};

export type GenderScore = {
  /** -1 masculine-coded … +1 feminine-coded. Acoustic, not identity. */
  score: number;
  label: GenderLabel;
  confidence: number;
  f0Hz: number;
  formantScale: number;
  cues: string[];
  /** Pitch-only cue, independent of the vowel. */
  pitchCue: number;
  /** Resonance/VTL cue from F3, mostly independent of the vowel. */
  resonanceCue: number;
  vtlCm: number;
  f3Hz: number;
};

export type AttributeScores = {
  brightness: number;
  breathiness: number;
  nasality: number;
  roughness: number;
  tension: number;
  resonance: number;
};

export type PhonemeHit = {
  id: string;
  ipa: string;
  arpabet: string;
  example: string;
  manner: Manner;
  start: number;
  end: number;
  confidence: number;
  volumeDb: number;
  gender: GenderScore;
  attributes: AttributeScores;
  subphonemes: Subphoneme[];
  formants: { f1: number; f2: number; f3: number };
  f0: number;
  word?: string;
};

export type WordHit = {
  word: string;
  start: number;
  end: number;
};

export type SpectrogramData = {
  data: Float32Array;
  rows: number;
  cols: number;
  binHz: number;
};

export type AnalysisResult = {
  duration: number;
  sampleRate: number;
  transcript: string;
  words: WordHit[];
  phonemes: PhonemeHit[];
  meanF0: number;
  meanVolumeDb: number;
  overallGender: GenderScore;
  meanF1: number;
  meanF2: number;
  meanF3: number;
  vtlCm: number;
  bassRatio: number;
  micNote: string | null;
  pitchRange: "chest" | "speech" | "head";
  backend: "webgpu" | "cpu";
  sourceName: string;
  spectrogram: SpectrogramData;
  waveform: Float32Array;
  formantTrack: { t: number; f1: number; f2: number; f3: number; f0: number }[];
};

export type TranscriptHint = {
  text: string;
  words?: WordHit[];
};
