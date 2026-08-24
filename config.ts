import path from "node:path";

export const ROOT = __dirname;

export const config = {
  paths: {
    weights: path.join(ROOT, "models", "weights.json"),
    vocab: path.join(ROOT, "models", "vocab.json"),
    knowledge: path.join(ROOT, "data", "knowledge.json"),
    memory: path.join(ROOT, "data", "memory.json"),
    settings: path.join(ROOT, "data", "settings.json"),
    training: path.join(ROOT, "data", "training.json")
  },
  model: {
    maxVocab: 4096,
    embeddingSize: 64,
    hiddenSize: 128,
    contextSize: 12,
    temperature: 0.75,
    topK: 8
  },
  training: {
    learningRate: 0.05,
    epochsPerRun: 5,
    stepsPerExample: 32,
    auto: true,
    saveEvery: 1
  }
};
