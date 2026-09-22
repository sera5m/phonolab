import { resampleLinear } from "./wav";
import { TARGET_SR } from "./dsp";

let sharedCtx: AudioContext | null = null;

export function getAudioContext() {
  if (typeof window === "undefined") {
    throw new Error("Audio runs in the browser only");
  }
  if (!sharedCtx || sharedCtx.state === "closed") {
    sharedCtx = new AudioContext();
  }
  return sharedCtx;
}

export async function decodeFile(file: File) {
  const buf = await file.arrayBuffer();
  return decodeArrayBuffer(buf, file.name);
}

export async function decodeArrayBuffer(buf: ArrayBuffer, name: string) {
  const ctx = getAudioContext();
  if (ctx.state === "suspended") await ctx.resume();
  const audio = await ctx.decodeAudioData(buf.slice(0));
  const mixed = mixdown(audio);
  const samples = resampleLinear(mixed, audio.sampleRate, TARGET_SR);
  const maxSeconds = 45;
  const maxN = TARGET_SR * maxSeconds;
  const clipped = samples.length > maxN ? samples.subarray(0, maxN) : samples;
  return {
    samples: new Float32Array(clipped),
    sampleRate: TARGET_SR,
    duration: clipped.length / TARGET_SR,
    name,
    channels: audio.numberOfChannels,
    originalRate: audio.sampleRate,
  };
}

function mixdown(buffer: AudioBuffer) {
  const n = buffer.length;
  const out = new Float32Array(n);
  const ch = buffer.numberOfChannels;
  for (let c = 0; c < ch; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < n; i++) out[i]! += data[i]!;
  }
  if (ch > 1) {
    for (let i = 0; i < n; i++) out[i]! /= ch;
  }
  return out;
}

export async function playSamples(samples: Float32Array, sampleRate: number, start = 0, end?: number) {
  const ctx = getAudioContext();
  if (ctx.state === "suspended") await ctx.resume();
  const i0 = Math.max(0, Math.floor(start * sampleRate));
  const i1 = Math.min(samples.length, Math.floor((end ?? samples.length / sampleRate) * sampleRate));
  const slice = samples.subarray(i0, Math.max(i0 + 1, i1));
  const buffer = ctx.createBuffer(1, slice.length, sampleRate);
  const copy = new Float32Array(slice.length);
  copy.set(slice);
  buffer.copyToChannel(copy, 0);
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(ctx.destination);
  src.start();
  return src;
}
