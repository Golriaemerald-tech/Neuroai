# NeuroAI — Node.js + TSX

This is the new Node/TypeScript/React-TSX version of NeuroAI.

## Run on your phone with Termux

```bash
pkg update
pkg install nodejs
cd /storage/emulated/0/NeuroAi
npm install
npm run build
npm start
```

Open `http://localhost:3000`.

For development with live Vite UI:

```bash
npm run dev
```

The native Node server uses **no Express**. The browser UI is React TSX. The backend is TypeScript executed by `tsx`.

## Project structure

- `src/App.tsx` — full UI
- `src/main.tsx` — React entry
- `src/styles.css` — UI styling
- `server.ts` — native Node HTTP server
- `brain.ts` — retrieval + response logic
- `model.ts` — neural model + training math
- `trainer.ts` — automatic training loop
- `memory.ts` — persistent memory
- `tokenizer.ts` — vocabulary/tokenization
- `data/knowledge.json` — knowledge database
- `models/weights.json` — learned parameters

No Express, Gemini, Google Search, or third-party AI API is required.
