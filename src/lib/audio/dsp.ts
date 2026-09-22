import { clamp } from "@/lib/utils";
import { fft, hamming, nextPow2, preEmphasis } from "./fft";

export const TARGET_SR = 16000;
export const FRAME_MS = 25;
export const HOP_MS = 10;

export type FrameFeatures = {
  t: number;
  rms: number;
  db: number;
  zcr: number;
  f0: number;
  f0Prob: number;
  voiced: boolean;
  f1: number;
  f2: number;
  f3: number;
  centroid: number;
  rolloff: number;
  flux: number;
  hnr: number;
  spectrum: Float32Array;
};

export type Track = {
  sampleRate: number;
  hop: number;
  fftSize: number;
  frames: FrameFeatures[];
  samples: Float32Array;
  duration: number;
};

export function frameParams(sr: number) {
  const hop = Math.round((HOP_MS / 1000) * sr);
  const frame = Math.round((FRAME_MS / 1000) * sr);
  const fftSize = nextPow2(Math.max(256, frame));
  return { hop, frame, fftSize };
}

export async function extractTrack(samples: Float32Array, sampleRate: number): Promise<Track> {
  const { hop, frame, fftSize } = frameParams(sampleRate);
  const window = hamming(frame);
  const emphasized = preEmphasis(samples);
  const nyquist = sampleRate / 2;
  const binHz = sampleRate / fftSize;
  const frames: FrameFeatures[] = [];
  let prevMag: Float32Array | null = null;

  for (let start = 0; start + frame <= emphasized.length; start += hop) {
    const slice = emphasized.subarray(start, start + frame);
    const raw = samples.subarray(start, start + frame);
    const re = new Float32Array(fftSize);
    const im = new Float32Array(fftSize);
    for (let i = 0; i < frame; i++) re[i] = (slice[i] ?? 0) * (window[i] ?? 0);
    fft(re, im);

    const mag = new Float32Array(fftSize / 2);
    let energy = 0;
    for (let k = 0; k < mag.length; k++) {
      const m = Math.hypot(re[k] ?? 0, im[k] ?? 0);
      mag[k] = m;
      energy += m * m;
    }

    let zc = 0;
    for (let i = 1; i < raw.length; i++) {
      if ((raw[i]! >= 0) !== (raw[i - 1]! >= 0)) zc++;
    }
    const zcr = zc / raw.length;

    let sum = 0;
    for (let i = 0; i < raw.length; i++) sum += raw[i]! * raw[i]!;
    const rms = Math.sqrt(sum / raw.length);
    const db = 20 * Math.log10(Math.max(rms, 1e-8));

    let weighted = 0;
    let magSum = 0;
    let rolloffEnergy = 0;
    const targetRolloff = energy * 0.85;
    let rolloff = nyquist;
    for (let k = 1; k < mag.length; k++) {
      const m = mag[k]!;
      magSum += m;
      weighted += m * k * binHz;
      rolloffEnergy += m * m;
      if (rolloff === nyquist && rolloffEnergy >= targetRolloff) rolloff = k * binHz;
    }
    const centroid = magSum > 0 ? weighted / magSum : 0;

    let flux = 0;
    if (prevMag) {
      for (let k = 0; k < mag.length; k++) {
        const d = mag[k]! - prevMag[k]!;
        if (d > 0) flux += d;
      }
    }
    prevMag = mag;

    const pitch = yinF0(raw, sampleRate);
    const formants = estimateFormants(slice, sampleRate, fftSize);
    const hnr = estimateHnr(raw, pitch.f0, sampleRate);
    const voiced = pitch.prob > 0.45 && db > -45 && zcr < 0.18;

    frames.push({
      t: start / sampleRate,
      rms,
      db,
      zcr,
      f0: voiced ? pitch.f0 : 0,
      f0Prob: pitch.prob,
      voiced,
      f1: formants[0] ?? 0,
      f2: formants[1] ?? 0,
      f3: formants[2] ?? 0,
      centroid,
      rolloff,
      flux,
      hnr,
      spectrum: mag,
    });
    if (frames.length % 48 === 0) {
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  return {
    sampleRate,
    hop,
    fftSize,
    frames,
    samples,
    duration: samples.length / sampleRate,
  };
}

export function spectrogramMatrix(track: Track, maxBins = 256) {
  const bins = Math.min(maxBins, Math.floor(track.fftSize / 2));
  const rows = bins;
  const cols = track.frames.length;
  const data = new Float32Array(rows * cols);
  let peak = 1e-8;
  for (const frame of track.frames) {
    for (let k = 0; k < bins; k++) peak = Math.max(peak, frame.spectrum[k] ?? 0);
  }
  const logPeak = Math.log10(peak + 1e-12);
  track.frames.forEach((frame, x) => {
    for (let y = 0; y < rows; y++) {
      const mag = frame.spectrum[y] ?? 0;
      const db = Math.log10(mag + 1e-12) - logPeak;
      const norm = clamp((db + 5) / 5, 0, 1);
      data[x + (rows - 1 - y) * cols] = norm;
    }
  });
  return { data, rows, cols, binHz: track.sampleRate / track.fftSize };
}

function yinF0(frame: Float32Array, sr: number, minF = 70, maxF = 420) {
  const n = frame.length;
  const tauMax = Math.min(n - 2, Math.floor(sr / minF));
  const tauMin = Math.max(2, Math.floor(sr / maxF));
  if (tauMax <= tauMin + 2) return { f0: 0, prob: 0 };

  const d = new Float32Array(tauMax + 1);
  for (let tau = 1; tau <= tauMax; tau++) {
    let sum = 0;
    const limit = n - tau;
    for (let j = 0; j < limit; j++) {
      const diff = frame[j]! - frame[j + tau]!;
      sum += diff * diff;
    }
    d[tau] = sum;
  }
  const cmnd = new Float32Array(tauMax + 1);
  cmnd[0] = 1;
  let running = 0;
  for (let tau = 1; tau <= tauMax; tau++) {
    running += d[tau]!;
    cmnd[tau] = running > 0 ? (d[tau]! * tau) / running : 1;
  }
  const thresh = 0.14;
  let tauEst = -1;
  for (let tau = tauMin; tau < tauMax; tau++) {
    if (cmnd[tau]! < thresh) {
      while (tau + 1 < tauMax && cmnd[tau + 1]! < cmnd[tau]!) tau++;
      tauEst = tau;
      break;
    }
  }
  if (tauEst < 0) {
    let best = tauMin;
    for (let tau = tauMin; tau <= tauMax; tau++) {
      if (cmnd[tau]! < cmnd[best]!) best = tau;
    }
    if (cmnd[best]! > 0.45) return { f0: 0, prob: 0 };
    tauEst = best;
  }
  const s0 = cmnd[tauEst - 1] ?? cmnd[tauEst]!;
  const s1 = cmnd[tauEst]!;
  const s2 = cmnd[tauEst + 1] ?? s1;
  const denom = 2 * (2 * s1 - s2 - s0);
  const shift = denom !== 0 ? (s2 - s0) / denom : 0;
  const tau = tauEst + shift;
  const f0 = sr / tau;
  const prob = clamp(1 - s1, 0, 1);
  return { f0, prob };
}

function estimateFormants(frame: Float32Array, sr: number, fftSize: number) {
  const order = sr > 12000 ? 14 : 10;
  const a = lpc(frame, order);
  if (!a) return [0, 0, 0];
  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);
  re[0] = 1;
  for (let i = 0; i < a.length; i++) re[i + 1] = a[i] ?? 0;
  fft(re, im);
  const env = new Float32Array(fftSize / 2);
  for (let k = 0; k < env.length; k++) {
    const mag = Math.hypot(re[k] ?? 0, im[k] ?? 0);
    env[k] = mag > 1e-12 ? 1 / mag : 0;
  }
  const binHz = sr / fftSize;
  const minBin = Math.round(180 / binHz);
  const maxBin = Math.round(Math.min(4200, sr / 2 - 100) / binHz);
  const peaks: { k: number; v: number }[] = [];
  for (let k = minBin + 1; k < maxBin; k++) {
    const v = env[k]!;
    if (v > (env[k - 1] ?? 0) && v >= (env[k + 1] ?? 0) && v > 0) {
      peaks.push({ k, v });
    }
  }
  peaks.sort((p, q) => q.v - p.v);
  const chosen: number[] = [];
  for (const p of peaks) {
    const hz = p.k * binHz;
    if (chosen.every((c) => Math.abs(c - hz) > 280)) {
      chosen.push(hz);
      if (chosen.length >= 3) break;
    }
  }
  chosen.sort((a, b) => a - b);
  while (chosen.length < 3) chosen.push(0);
  return chosen;
}

function lpc(frame: Float32Array, order: number) {
  const n = frame.length;
  const r = new Float32Array(order + 1);
  for (let lag = 0; lag <= order; lag++) {
    let sum = 0;
    for (let i = 0; i < n - lag; i++) sum += frame[i]! * frame[i + lag]!;
    r[lag] = sum;
  }
  if (r[0]! <= 1e-12) return null;
  const a = new Float32Array(order);
  const e = new Float32Array(order + 1);
  e[0] = r[0]!;
  for (let i = 0; i < order; i++) {
    let acc = r[i + 1]!;
    for (let j = 0; j < i; j++) acc += a[j]! * r[i - j]!;
    const ki = -acc / e[i]!;
    const next = a.slice();
    next[i] = ki;
    for (let j = 0; j < i; j++) next[j] = a[j]! + ki * a[i - 1 - j]!;
    for (let j = 0; j <= i; j++) a[j] = next[j]!;
    e[i + 1] = e[i]! * (1 - ki * ki);
    if (e[i + 1]! <= 1e-12) break;
  }
  return a;
}

function estimateHnr(frame: Float32Array, f0: number, sr: number) {
  if (f0 < 60) return 0;
  const period = Math.round(sr / f0);
  if (period < 4 || period >= frame.length / 2) return 0;
  let r0 = 0;
  let rT = 0;
  const n = frame.length - period;
  for (let i = 0; i < n; i++) {
    r0 += frame[i]! * frame[i]!;
    rT += frame[i]! * frame[i + period]!;
  }
  const noise = Math.max(r0 - rT, 1e-12);
  return 10 * Math.log10(Math.max(rT, 1e-12) / noise);
}

export function mean(values: number[]) {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function stdev(values: number[]) {
  if (values.length < 2) return 0;
  const m = mean(values);
  const v = values.reduce((a, b) => a + (b - m) ** 2, 0) / (values.length - 1);
  return Math.sqrt(v);
}

export function slope(values: number[]) {
  if (values.length < 2) return 0;
  const n = values.length;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;
  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += values[i]!;
    sumXY += i * values[i]!;
    sumXX += i * i;
  }
  const denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 1e-9) return 0;
  return (n * sumXY - sumX * sumY) / denom;
}
