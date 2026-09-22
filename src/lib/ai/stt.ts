import { createServerFn } from "@tanstack/react-start";

export type SttWord = { word: string; start: number; end: number };

export type SttResult =
  | { ok: true; text: string; words: SttWord[] }
  | { ok: false; error: string };

export const transcribeClip = createServerFn({ method: "POST" })
  .validator((input: { wavBase64: string; filename: string }) => input)
  .handler(async ({ data }): Promise<SttResult> => {
    const apiKey = process.env.XAI_API_KEY;
    if (!apiKey) return { ok: false, error: "Voice transcription is not available in this environment." };
    if (!data.wavBase64 || data.wavBase64.length > 4_500_000) {
      return { ok: false, error: "Clip is too large. Keep it under ~30 seconds." };
    }

    let binary: Uint8Array;
    try {
      binary = Buffer.from(data.wavBase64, "base64");
    } catch {
      return { ok: false, error: "Could not read the audio payload." };
    }

    const form = new FormData();
    form.append("model", "grok-voice-transcribe-2.0");
    form.append("file", new Blob([new Uint8Array(binary)], { type: "audio/wav" }), data.filename || "clip.wav");

    const res = await fetch("https://api.x.ai/v1/stt", {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}` },
      body: form,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      return { ok: false, error: `Transcription failed (${res.status}). ${detail.slice(0, 180)}` };
    }
    const body = (await res.json()) as Record<string, unknown>;
    const text = pickText(body);
    const words = pickWords(body);
    if (!text) return { ok: false, error: "The transcriber returned empty text." };
    return { ok: true, text, words };
  });

export const interpretVoice = createServerFn({ method: "POST" })
  .validator((input: { summary: string }) => input)
  .handler(async ({ data }): Promise<{ ok: true; text: string } | { ok: false; error: string }> => {
    const apiKey = process.env.XAI_API_KEY;
    if (!apiKey) return { ok: false, error: "AI notes are not available in this environment." };
    const res = await fetch("https://api.x.ai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "grok-4.5",
        max_tokens: 420,
        messages: [
          {
            role: "system",
            content:
              "You are a phonetician. Interpret the acoustic analysis. Be precise, not mystical. Gender scores are acoustic (F0 + formant scale), not identity. Mention onset/nucleus/coda contours when relevant. Short paragraphs.",
          },
          { role: "user", content: data.summary.slice(0, 6000) },
        ],
      }),
    });
    if (!res.ok) return { ok: false, error: `xAI API error ${res.status}` };
    const body = (await res.json()) as { choices?: { message?: { content?: string } }[] };
    return { ok: true, text: body.choices?.[0]?.message?.content ?? "" };
  });

function pickText(body: Record<string, unknown>) {
  if (typeof body.text === "string") return body.text.trim();
  if (typeof body.transcript === "string") return body.transcript.trim();
  const nested = body.result as Record<string, unknown> | undefined;
  if (nested && typeof nested.text === "string") return nested.text.trim();
  return "";
}

function pickWords(body: Record<string, unknown>): SttWord[] {
  const raw =
    (body.words as unknown[]) ||
    (body.segments as unknown[]) ||
    ((body.result as Record<string, unknown> | undefined)?.words as unknown[]) ||
    [];
  if (!Array.isArray(raw)) return [];
  const words: SttWord[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const rec = item as Record<string, unknown>;
    const word = String(rec.word ?? rec.text ?? "").trim();
    const start = Number(rec.start ?? rec.start_time ?? rec.begin ?? 0);
    const end = Number(rec.end ?? rec.end_time ?? rec.stop ?? start);
    if (word) words.push({ word, start, end });
  }
  return words;
}
