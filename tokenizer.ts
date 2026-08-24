import fs from "node:fs";
import { config } from "./config";

type Vocab = { word2idx: Record<string, number>; idx2word: Record<string, string> };
const fallback: Vocab = { word2idx: {"<pad>":0,"<eos>":1,"<unk>":2}, idx2word: {"0":"<pad>","1":"<eos>","2":"<unk>"} };

export function loadVocab(): Vocab {
  try {
    const raw = JSON.parse(fs.readFileSync(config.paths.vocab, "utf8"));
    if (raw.word2idx && raw.idx2word) return raw;
  } catch {}
  return fallback;
}

export function tokenize(text: string): string[] {
  return String(text || "").toLowerCase()
    .replace(/([!?.,;:])/g, " $1 ")
    .split(/\s+/).map(x => x.trim()).filter(Boolean);
}

export function vocabSize(): number {
  return Object.keys(loadVocab().word2idx).length;
}

export function learnVocabulary(text: string) {
  const v = loadVocab();
  for (const token of tokenize(text)) {
    if (v.word2idx[token] === undefined && Object.keys(v.word2idx).length < config.model.maxVocab) {
      const id = Object.keys(v.word2idx).length;
      v.word2idx[token] = id;
      v.idx2word[String(id)] = token;
    }
  }
  fs.mkdirSync(require("node:path").dirname(config.paths.vocab), { recursive: true });
  fs.writeFileSync(config.paths.vocab, JSON.stringify(v), "utf8");
}
