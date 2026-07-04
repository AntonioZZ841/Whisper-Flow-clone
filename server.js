// Whisper Flow clone — server.
//
// Per-utterance pipeline, mirroring the real product's two-stage design:
//
//   audio blob → Stage 1  ASR (Whisper API)  → raw transcript
//              → Stage 2  Flow (LLM cleanup) → formatted text
//
import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { transcribe, asrConfig } from './src/asr.js';
import { flow, flowConfig } from './src/flow.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;

const app = express();

// Clips are short, single utterances — keep them in memory and forward
// straight to the ASR provider; nothing touches disk.
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 }, // Whisper's own upload cap
});

app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, asr: asrConfig(), flow: flowConfig() });
});

app.post('/api/transcribe', upload.single('audio'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No audio file was uploaded.' });
  }
  try {
    const asr = await transcribe(req.file);
    const flowed = await flow(asr.text);
    res.json({
      raw: asr.text,
      formatted: flowed.text,
      meta: {
        asr: { mock: asr.mock },
        flow: {
          provider: flowed.provider,
          model: flowed.model,
          ...(flowed.error ? { error: flowed.error } : {}),
        },
      },
    });
  } catch (err) {
    console.error('Transcription failed:', err);
    res
      .status(502)
      .json({ error: 'Transcription failed.', detail: String(err?.message ?? err) });
  }
});

export { app };

// Only start listening when run directly (`node server.js`) — importing the
// app for tests must not bind a port.
const isDirectRun = process.argv[1] === fileURLToPath(import.meta.url);
if (isDirectRun) {
  app.listen(PORT, () => {
    const a = asrConfig();
    const f = flowConfig();
    console.log(`Whisper Flow clone → http://localhost:${PORT}`);
    console.log(
      `  Stage 1 ASR:  ${a.provider}${a.model ? ` (${a.model})` : ''}` +
        `${a.gpu ? ' · GPU detected' : ''}${a.reason ? ` · ${a.reason}` : ''}`,
    );
    console.log(
      `  Stage 2 Flow: ${f.provider === 'passthrough' ? 'passthrough (no LLM key set)' : `${f.provider} (${f.model})`}`,
    );
  });
}
