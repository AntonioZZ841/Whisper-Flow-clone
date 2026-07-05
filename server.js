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
import https from 'node:https';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { transcribe, asrConfig } from './src/asr.js';
import { flow, flowConfig } from './src/flow.js';
import { getLanIPv4s } from './src/net.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
// 0.0.0.0 (the default) binds every network interface, not just loopback —
// this is what makes the server reachable from other devices on the LAN.
// Narrow it (e.g. to 127.0.0.1) if you want to force localhost-only.
const HOST = process.env.HOST || '0.0.0.0';

// Browsers only allow mic capture (getUserMedia) in a "secure context" —
// localhost is exempt, but any other origin (e.g. a phone on the same LAN)
// needs HTTPS. `npm run cert` writes these; their presence is what switches
// the dev server from HTTP to HTTPS, no separate flag to remember.
const HTTPS_KEY_FILE = process.env.HTTPS_KEY_FILE || path.join(__dirname, 'certs', 'dev-key.pem');
const HTTPS_CERT_FILE = process.env.HTTPS_CERT_FILE || path.join(__dirname, 'certs', 'dev-cert.pem');
const httpsAvailable = existsSync(HTTPS_KEY_FILE) && existsSync(HTTPS_CERT_FILE);

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
  const onListening = () => {
    const scheme = httpsAvailable ? 'https' : 'http';
    const a = asrConfig();
    const f = flowConfig();
    console.log(`Whisper Flow clone → ${scheme}://localhost:${PORT}`);
    if (httpsAvailable) {
      for (const ip of getLanIPv4s()) console.log(`  also reachable at  ${scheme}://${ip}:${PORT}`);
    } else {
      console.log(
        '  Mic capture needs a secure context off localhost — run `npm run cert` for HTTPS on your LAN.',
      );
    }
    console.log(
      `  Stage 1 ASR:  ${a.provider}${a.model ? ` (${a.model})` : ''}` +
        `${a.gpu ? ' · GPU detected' : ''}${a.reason ? ` · ${a.reason}` : ''}`,
    );
    console.log(
      `  Stage 2 Flow: ${f.provider === 'passthrough' ? 'passthrough (no LLM key set)' : `${f.provider} (${f.model})`}`,
    );
  };

  if (httpsAvailable) {
    https
      .createServer(
        { key: readFileSync(HTTPS_KEY_FILE), cert: readFileSync(HTTPS_CERT_FILE) },
        app,
      )
      .listen(PORT, HOST, onListening);
  } else {
    app.listen(PORT, HOST, onListening);
  }
}
