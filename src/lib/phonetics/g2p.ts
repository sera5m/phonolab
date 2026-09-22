import { ARPA_TO_IPA, CMU } from "./data";
import type { Manner } from "./types";

export type G2PPhone = {
  ipa: string;
  arpabet: string;
  manner: Manner;
  example: string;
  word: string;
};

export function tokenizeWords(text: string) {
  return text
    .toLowerCase()
    .replace(/[^a-z'\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

export function wordToPhones(word: string): G2PPhone[] {
  const key = word.toLowerCase().replace(/'/g, "");
  const arpa = CMU[key] ?? guessArpa(key);
  return arpa.map((a) => {
    const meta = ARPA_TO_IPA[a] ?? { ipa: a.toLowerCase(), manner: "vowel" as const, example: word };
    return { ipa: meta.ipa, arpabet: a, manner: meta.manner, example: meta.example, word };
  });
}

export function textToPhones(text: string) {
  return tokenizeWords(text).flatMap(wordToPhones);
}

function guessArpa(word: string): string[] {
  const out: string[] = [];
  let i = 0;
  const w = word;
  while (i < w.length) {
    const two = w.slice(i, i + 2);
    const three = w.slice(i, i + 3);
    if (three === "igh") {
      out.push("AY");
      i += 3;
      continue;
    }
    if (two === "ch") {
      out.push("CH");
      i += 2;
      continue;
    }
    if (two === "sh") {
      out.push("SH");
      i += 2;
      continue;
    }
    if (two === "th") {
      out.push(i === 0 ? "TH" : "DH");
      i += 2;
      continue;
    }
    if (two === "ng") {
      out.push("NG");
      i += 2;
      continue;
    }
    if (two === "oo") {
      out.push("UW");
      i += 2;
      continue;
    }
    if (two === "ee") {
      out.push("IY");
      i += 2;
      continue;
    }
    if (two === "ay" || two === "ai") {
      out.push("EY");
      i += 2;
      continue;
    }
    if (two === "ow") {
      out.push("OW");
      i += 2;
      continue;
    }
    if (two === "ou") {
      out.push("AW");
      i += 2;
      continue;
    }
    if (two === "er" || two === "ir" || two === "ur") {
      out.push("ER");
      i += 2;
      continue;
    }
    const c = w[i]!;
    const map: Record<string, string> = {
      a: "AE",
      e: "EH",
      i: "IH",
      o: "AA",
      u: "AH",
      y: "IY",
      b: "B",
      c: "K",
      d: "D",
      f: "F",
      g: "G",
      h: "HH",
      j: "JH",
      k: "K",
      l: "L",
      m: "M",
      n: "N",
      p: "P",
      q: "K",
      r: "R",
      s: "S",
      t: "T",
      v: "V",
      w: "W",
      x: "K",
      z: "Z",
    };
    if (c === "x") out.push("K", "S");
    else if (map[c]) out.push(map[c]);
    i += 1;
  }
  return out.length ? out : ["AH"];
}
