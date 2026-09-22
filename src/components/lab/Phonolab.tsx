import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronDown,
  Cpu,
  Loader2,
  Mic,
  Square,
  Upload,
  AudioLines,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { DEMO_CLIPS, synthesize, type DemoClip } from "@/lib/audio/synth";
import { decodeFile, playSamples } from "@/lib/audio/decode";
import { detectGpu, type GpuInfo } from "@/lib/audio/gpu";
import { wavToBase64 } from "@/lib/audio/wav";
import { analyzeVoice } from "@/lib/phonetics/pipeline";
import { interpretVoice, transcribeClip } from "@/lib/ai/stt";
import type { AnalysisResult, PhonemeHit } from "@/lib/phonetics/types";
import { cn, formatDb, formatHz, formatPct } from "@/lib/utils";
import { Spectrogram } from "./Spectrogram";
import { PhonemeTimeline } from "./PhonemeTimeline";
import { PhonemeDetail } from "./PhonemeDetail";

type Status = "boot" | "ready" | "working" | "error";

const SAMPLE_CACHE: { samples: Float32Array; sampleRate: number } = {
  samples: new Float32Array(0),
  sampleRate: 16000,
};

export function Phonolab() {
  const [gpu, setGpu] = useState<GpuInfo | null>(null);
  const [status, setStatus] = useState<Status>("boot");
  const [stage, setStage] = useState("Warming analyser");
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);
  const [recording, setRecording] = useState(false);
  const [aiNote, setAiNote] = useState<string | null>(null);
  const [aiBusy, setAiBusy] = useState<"stt" | "note" | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const didBoot = useRef(false);

  const selected: PhonemeHit | null = useMemo(() => {
    if (!analysis) return null;
    return analysis.phonemes.find((p) => p.id === selectedId) ?? analysis.phonemes[0] ?? null;
  }, [analysis, selectedId]);

  useEffect(() => {
    detectGpu().then(setGpu);
  }, []);

  const runAnalysis = useCallback(
    async (opts: {
      samples: Float32Array;
      sampleRate: number;
      name: string;
      known?: { ipa: string; start: number; end: number; spec?: DemoClip["phonemes"][number] }[];
      transcript?: { text: string };
    }) => {
      setStatus("working");
      setStage("Framing · STFT · pitch · LPC");
      setError(null);
      setAiNote(null);
      SAMPLE_CACHE.samples = opts.samples;
      SAMPLE_CACHE.sampleRate = opts.sampleRate;
      try {
        await new Promise((r) => setTimeout(r, 30));
        const result = await analyzeVoice({
          samples: opts.samples,
          sampleRate: opts.sampleRate,
          sourceName: opts.name,
          backend: gpu?.backend ?? "cpu",
          known: opts.known,
          transcript: opts.transcript,
        });
        setAnalysis(result);
        setSelectedId(result.phonemes[0]?.id ?? null);
        setStatus("ready");
      } catch (err) {
        setStatus("error");
        setError(err instanceof Error ? err.message : "Analysis failed");
      }
    },
    [gpu],
  );

  const loadDemo = useCallback(
    async (clip: DemoClip) => {
      setStage(`Synthesizing ${clip.name}`);
      setStatus("working");
      const synth = synthesize(clip.phonemes);
      await runAnalysis({
        samples: synth.samples,
        sampleRate: synth.sampleRate,
        name: clip.name,
        known: synth.timeline.map((t) => ({
          ipa: t.ipa,
          start: t.start,
          end: t.end,
          spec: t.spec,
        })),
        transcript: { text: clip.transcript },
      });
    },
    [runAnalysis],
  );

  useEffect(() => {
    if (didBoot.current) return;
    didBoot.current = true;
    const boot = DEMO_CLIPS[0];
    if (boot) void loadDemo(boot);
  }, [loadDemo]);

  async function onFile(file: File) {
    setStatus("working");
    setStage("Decoding clip");
    try {
      const decoded = await decodeFile(file);
      await runAnalysis({
        samples: decoded.samples,
        sampleRate: decoded.sampleRate,
        name: file.name,
      });
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Could not decode that file");
    }
  }

  async function toggleRecord() {
    if (recording) {
      recorderRef.current?.stop();
      setRecording(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size) chunksRef.current.push(e.data);
      };
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
        const file = new File([blob], "recording.webm", { type: blob.type });
        await onFile(file);
      };
      recorderRef.current = rec;
      rec.start();
      setRecording(true);
      setTimeout(() => {
        if (recorderRef.current === rec && rec.state === "recording") rec.stop();
        setRecording(false);
      }, 15000);
    } catch {
      setError("Microphone permission was denied.");
      setStatus("error");
    }
  }

  async function transcribe() {
    if (!SAMPLE_CACHE.samples.length) return;
    setAiBusy("stt");
    setError(null);
    try {
      const wavBase64 = wavToBase64(SAMPLE_CACHE.samples, SAMPLE_CACHE.sampleRate);
      const result = await transcribeClip({
        data: { wavBase64, filename: `${analysis?.sourceName ?? "clip"}.wav` },
      });
      if (!result.ok) {
        setError(result.error);
        return;
      }
      await runAnalysis({
        samples: SAMPLE_CACHE.samples,
        sampleRate: SAMPLE_CACHE.sampleRate,
        name: analysis?.sourceName ?? "clip",
        transcript: { text: result.text, ...(result.words.length ? { words: result.words } : {}) },
      });
    } finally {
      setAiBusy(null);
    }
  }

  async function askNote() {
    if (!analysis) return;
    setAiBusy("note");
    setError(null);
    const summary = [
      `Source: ${analysis.sourceName}`,
      `Transcript: ${analysis.transcript}`,
      `Mean F0 ${Math.round(analysis.meanF0)} Hz, volume ${analysis.meanVolumeDb.toFixed(1)} dB`,
      `Overall gender: ${analysis.overallGender.label} (${analysis.overallGender.score.toFixed(2)}, conf ${analysis.overallGender.confidence.toFixed(2)})`,
      ...analysis.phonemes.map(
        (p) =>
          `/${p.ipa}/ ${p.start.toFixed(2)}–${p.end.toFixed(2)}s F0=${Math.round(p.f0)} F1=${Math.round(p.formants.f1)} F2=${Math.round(p.formants.f2)} vol=${p.volumeDb.toFixed(1)} gender=${p.gender.score.toFixed(2)} ${p.subphonemes.map((s) => `${s.phase}:${s.contour}`).join(",")}`,
      ),
    ].join("\n");
    const result = await interpretVoice({ data: { summary } });
    setAiBusy(null);
    if (!result.ok) setError(result.error);
    else setAiNote(result.text);
  }

  function playSelection() {
    if (!selected || !SAMPLE_CACHE.samples.length) return;
    void playSamples(SAMPLE_CACHE.samples, SAMPLE_CACHE.sampleRate, selected.start, selected.end);
  }

  function playAll() {
    if (!SAMPLE_CACHE.samples.length) return;
    void playSamples(SAMPLE_CACHE.samples, SAMPLE_CACHE.sampleRate);
  }

  function selectAtTime(t: number) {
    if (!analysis) return;
    const hit = analysis.phonemes.find((p) => t >= p.start && t < p.end) ?? analysis.phonemes[0];
    if (hit) setSelectedId(hit.id);
  }

  function exportJson() {
    if (!analysis) return;
    const blob = new Blob(
      [
        JSON.stringify(
          {
            source: analysis.sourceName,
            transcript: analysis.transcript,
            meanF0: analysis.meanF0,
            gender: analysis.overallGender,
            phonemes: analysis.phonemes.map((p) => ({
              ipa: p.ipa,
              arpabet: p.arpabet,
              start: p.start,
              end: p.end,
              confidence: p.confidence,
              volumeDb: p.volumeDb,
              gender: p.gender,
              attributes: p.attributes,
              formants: p.formants,
              subphonemes: p.subphonemes,
            })),
          },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "phonolab-analysis.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <TooltipProvider delayDuration={200}>
      <div className="min-h-dvh bg-bg text-fg">
        <header className="border-b border-border">
          <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-5 sm:flex-row sm:items-end sm:justify-between sm:px-6">
            <div>
              <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-subtle">
                Voice laboratory
              </p>
              <h1 className="font-display text-4xl tracking-tight text-fg text-balance sm:text-5xl">
                Phonolab
              </h1>
              <p className="mt-2 max-w-xl text-sm text-muted text-pretty">
                Pull phonemes out of a voice clip, score each one for acoustic gender, attributes, and
                volume, then inspect the spectrogram.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <GpuChip gpu={gpu} />
              <input
                ref={fileRef}
                type="file"
                accept="audio/*,.wav,.mp3,.ogg,.m4a,.webm,.flac"
                className="sr-only"
                tabIndex={-1}
                aria-hidden="true"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void onFile(f);
                  e.target.value = "";
                }}
              />
              <Button variant="secondary" onClick={() => fileRef.current?.click()}>
                <Upload />
                Open clip
              </Button>
              <Button variant={recording ? "danger" : "outline"} onClick={() => void toggleRecord()}>
                {recording ? <Square /> : <Mic />}
                {recording ? "Stop" : "Record"}
              </Button>
            </div>
          </div>
        </header>

        <main className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-6 sm:px-6 [overflow-anchor:none]">
          <section className="rounded-xl bg-surface p-3 shadow-[var(--shadow-border)] sm:p-4">
            <p className="mb-3 px-1 text-xs text-muted">Formant-synthesized references — known IPA, measured acoustics</p>
            <div className="grid gap-2 sm:grid-cols-2">
              {DEMO_CLIPS.map((clip) => (
                <button
                  key={clip.id}
                  type="button"
                  onClick={() => void loadDemo(clip)}
                  className="rounded-lg bg-surface-2 px-4 py-3 text-left shadow-[var(--shadow-border)] transition-[box-shadow] duration-150 hover:shadow-[var(--shadow-border-hover)]"
                >
                  <div className="font-medium text-fg">{clip.name}</div>
                  <div className="mt-1 text-xs text-muted text-pretty">{clip.blurb}</div>
                </button>
              ))}
            </div>
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files[0];
                if (f) void onFile(f);
              }}
              className="mt-3 flex min-h-16 cursor-pointer items-center justify-center rounded-lg border border-dashed border-border px-4 py-3 text-center text-sm text-muted"
              onClick={() => fileRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") fileRef.current?.click();
              }}
              role="button"
              tabIndex={0}
            >
              Drop a wav, mp3, or m4a — analysis stays on this machine except optional Grok transcription
            </div>
          </section>

          {status === "working" || status === "boot" ? (
            <div className="flex items-center gap-3 rounded-xl bg-surface px-4 py-5 shadow-[var(--shadow-border)]">
              <Loader2 className="size-4 animate-spin text-accent" />
              <div>
                <div className="text-sm text-fg">{stage}</div>
                <div className="text-xs text-subtle">STFT · YIN pitch · LPC formants · subphoneme contours</div>
              </div>
            </div>
          ) : null}

          {error ? (
            <div className="rounded-xl bg-danger/10 px-4 py-3 text-sm text-danger">{error}</div>
          ) : null}

          {analysis ? (
            <>
              <section className="rounded-xl bg-surface p-3 shadow-[var(--shadow-border)] sm:p-5">
                <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h2 className="text-sm font-medium text-fg">{analysis.sourceName}</h2>
                    <p className="mt-0.5 font-mono text-xs text-muted">
                      {analysis.duration.toFixed(2)} s · {formatHz(analysis.meanF0)} mean F0 ·{" "}
                      {formatDb(analysis.meanVolumeDb)} · {analysis.phonemes.length} phonemes
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button variant="outline" size="sm" onClick={playAll}>
                      <AudioLines />
                      Play clip
                    </Button>
                    <Button variant="ghost" size="sm" onClick={() => setExpanded((v) => !v)}>
                      <ChevronDown className={cn("transition-transform duration-200", expanded && "rotate-180")} />
                      {expanded ? "Collapse spectrum" : "Expand spectrum"}
                    </Button>
                  </div>
                </div>
                <Spectrogram
                  analysis={analysis}
                  selected={selected}
                  expanded={expanded}
                  playhead={selected ? (selected.start + selected.end) / 2 : null}
                  onSelectTime={selectAtTime}
                />
                <div className="mt-4">
                  <PhonemeTimeline
                    phonemes={analysis.phonemes}
                    duration={analysis.duration}
                    selectedId={selected?.id ?? null}
                    onSelect={setSelectedId}
                  />
                </div>
              </section>

              <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
                <div className="rounded-xl bg-surface p-4 shadow-[var(--shadow-border)] sm:p-5">
                  <h2 className="text-sm font-medium text-fg">Transcript</h2>
                  <p className="mt-2 font-display text-2xl leading-snug text-fg text-pretty">
                    {analysis.transcript || "Acoustic labels only — transcribe to attach English words."}
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <Badge variant="accent">{analysis.overallGender.label}</Badge>
                    <Badge>{formatPct(analysis.overallGender.confidence)} overall conf.</Badge>
                    <Badge variant="steel">{analysis.backend === "webgpu" ? "WebGPU" : "CPU"} path</Badge>
                  </div>
                  <Separator className="my-4" />
                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={aiBusy !== null}
                      onClick={() => void transcribe()}
                    >
                      {aiBusy === "stt" ? <Loader2 className="animate-spin" /> : <Sparkles />}
                      Transcribe with Grok
                    </Button>
                    <Button variant="outline" size="sm" disabled={aiBusy !== null} onClick={() => void askNote()}>
                      {aiBusy === "note" ? <Loader2 className="animate-spin" /> : <Sparkles />}
                      Phonetic reading
                    </Button>
                    <Button variant="ghost" size="sm" onClick={exportJson}>
                      Export JSON
                    </Button>
                  </div>
                  {aiNote ? (
                    <p className="mt-4 text-sm text-muted text-pretty">{aiNote}</p>
                  ) : (
                    <p className="mt-4 text-xs text-subtle">
                      Transcription and the phonetic reading are optional, user-started calls. The spectrogram,
                      gender, attributes, and volume are measured locally.
                    </p>
                  )}
                </div>
                <div className="rounded-xl bg-surface p-4 shadow-[var(--shadow-border)] sm:p-5">
                  {selected ? <PhonemeDetail phoneme={selected} onPlay={playSelection} /> : null}
                </div>
              </section>
            </>
          ) : null}

          <section className="rounded-xl bg-surface p-4 shadow-[var(--shadow-border)] sm:p-5">
            <button
              type="button"
              className="flex w-full items-center justify-between text-left"
              onClick={() => setGuideOpen((v) => !v)}
            >
              <span className="text-sm font-medium text-fg">How a phoneme is built</span>
              <ChevronDown className={cn("size-4 text-muted transition-transform duration-200", guideOpen && "rotate-180")} />
            </button>
            {guideOpen ? (
              <div className="mt-4 grid gap-4 text-sm text-muted md:grid-cols-3">
                <GuideCard
                  title="Onset · start"
                  body="Energy rises and formants transition out of the previous sound. For a stop this is the burst plus voice-onset time; for a vowel it is the formant glide in."
                />
                <GuideCard
                  title="Nucleus · center"
                  body="The steady target. Vowel identity lives in F1 (tongue height) and F2 (front/back). Gender perception leans on F0 here, with formant scale as a second cue."
                />
                <GuideCard
                  title="Coda · end"
                  body="Energy drops and formants head toward the next phoneme. Rise vs. drop of F0 and intensity in this window is the contour we score."
                />
              </div>
            ) : (
              <p className="mt-2 text-xs text-subtle">
                Spelled phoneme. Each one has an onset, nucleus, and coda — not a single snapshot.
              </p>
            )}
          </section>
        </main>
      </div>
    </TooltipProvider>
  );
}

function GpuChip({ gpu }: { gpu: GpuInfo | null }) {
  const label = !gpu ? "Detecting GPU" : gpu.available ? gpu.adapterName : "CPU analyser";
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span>
          <Badge variant={gpu?.available ? "accent" : "default"} className="h-9 gap-1.5 rounded-sm px-3">
            {gpu?.available ? <AudioLines className="size-3.5" /> : <Cpu className="size-3.5" />}
            {label}
          </Badge>
        </span>
      </TooltipTrigger>
      <TooltipContent>
        {gpu?.note ?? "Checking for WebGPU (Vulkan). AMD RX 7700S is supported through Mesa RADV — no CUDA."}
      </TooltipContent>
    </Tooltip>
  );
}

function GuideCard({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-md bg-surface-2 px-3 py-3">
      <div className="text-xs font-medium uppercase tracking-wider text-accent">{title}</div>
      <p className="mt-2 text-pretty">{body}</p>
    </div>
  );
}
