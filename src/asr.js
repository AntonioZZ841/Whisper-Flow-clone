// Stage 1 — ASR (speech → raw transcript).
//
// Wispr Flow transcribes in the cloud with Whisper-family models and
// processes one utterance at a time ("wait, understand, then write what you
// meant") rather than streaming word-by-word. This module supports three
// transcription providers:
//
//   mock               no key/GPU — a canned disfluent utterance for demos
//   openai-compatible  any /audio/transcriptions endpoint (OpenAI, whisper.cpp,
//                      faster-whisper) via TRANSCRIPTION_API_KEY + _URL
//   nemo               local NVIDIA NeMo ASR (e.g. Parakeet/Canary) — only
//                      offered when an NVIDIA GPU is present on the server
//
// The provider is chosen by resolveAsr() (pure + unit-tested); override with
// TRANSCRIPTION_PROVIDER=auto | mock | openai-compatible | nemo.

import { spawn, execFileSync } from 'node:child_process';
import { openAsBlob } from 'node:fs';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const MOCK_TRANSCRIPT =
  'um so hey can you move our sync to monday no wait actually tuesday at uh 3 p.m. and um also like send the notes uh send the notes to everyone after';

// Keyless demo for meeting mode: a canned two-person exchange, disfluent like
// real ASR output so the Flow stage has something to clean per turn.
const MOCK_TURNS = [
  { speaker: 'Speaker 1', start: 0, end: 4.1, text: 'um so are we still on for the quarterly review on monday no wait actually thursday' },
  { speaker: 'Speaker 2', start: 4.4, end: 7.9, text: 'uh yeah thursday works for me can you um can you send the invite' },
  { speaker: 'Speaker 1', start: 8.2, end: 10.5, text: 'sure I will I will send it right after lunch' },
];

const DEFAULT_NEMO_MODEL = 'nvidia/parakeet-tdt-0.6b-v2';

// Detect an NVIDIA GPU via `nvidia-smi`. Cached — the answer can't change
// within a process lifetime.
let gpuCache;
export function hasNvidiaGpu() {
  if (gpuCache === undefined) {
    try {
      execFileSync('nvidia-smi', ['-L'], { stdio: 'ignore' });
      gpuCache = true;
    } catch {
      gpuCache = false;
    }
  }
  return gpuCache;
}

// Pure provider resolution — takes the environment and whether a GPU is
// present, returns the effective config. Kept side-effect-free so it can be
// unit-tested for every branch without a real GPU.
export function resolveAsr(env, gpu) {
  const explicit = (env.TRANSCRIPTION_PROVIDER || 'auto').trim().toLowerCase();
  let provider;
  let reason;

  if (explicit === 'nemo') {
    if (gpu) {
      provider = 'nemo';
    } else {
      // NeMo needs a GPU; degrade to the next best available provider.
      provider = env.TRANSCRIPTION_API_KEY ? 'openai-compatible' : 'mock';
      reason = 'NeMo requested but no NVIDIA GPU detected';
    }
  } else if (explicit === 'openai-compatible') {
    provider = 'openai-compatible';
  } else if (explicit === 'mock') {
    provider = 'mock';
  } else {
    // auto: a configured cloud key wins; otherwise use local NeMo when a GPU
    // is available; otherwise fall back to the keyless mock.
    if (env.TRANSCRIPTION_API_KEY) provider = 'openai-compatible';
    else if (gpu && env.NEMO_ENABLED !== 'false') provider = 'nemo';
    else provider = 'mock';
  }

  const model =
    provider === 'openai-compatible'
      ? env.TRANSCRIPTION_MODEL || 'whisper-1'
      : provider === 'nemo'
        ? env.NEMO_MODEL || DEFAULT_NEMO_MODEL
        : null;

  return { provider, model, gpu, ...(reason ? { reason } : {}) };
}

export function asrConfig() {
  return resolveAsr(process.env, hasNvidiaGpu());
}

export async function transcribe(file, { diarize = false } = {}) {
  const cfg = asrConfig();
  if (cfg.provider === 'mock') {
    if (diarize) {
      return {
        text: MOCK_TURNS.map((t) => `${t.speaker}: ${t.text}`).join('\n'),
        turns: MOCK_TURNS,
        mock: true,
        provider: 'mock',
      };
    }
    return { text: MOCK_TRANSCRIPT, mock: true, provider: 'mock' };
  }
  if (cfg.provider === 'nemo') {
    return transcribeWithNemo(file, cfg.model, { diarize });
  }
  const out = await transcribeOpenAICompatible(file, cfg.model);
  if (diarize) {
    // /audio/transcriptions has no diarization concept — be explicit rather
    // than silently dropping the request.
    out.turns = null;
    out.warning = 'speaker labeling requires the local NeMo provider — returning an unlabeled transcript';
  }
  return out;
}

// ── openai-compatible (OpenAI / whisper.cpp / faster-whisper) ───────────────

async function transcribeOpenAICompatible({ buffer, path: filePath, mimetype, originalname }, model) {
  const url =
    process.env.TRANSCRIPTION_API_URL ||
    'https://api.openai.com/v1/audio/transcriptions';

  // Disk-spooled uploads (the normal case) become a file-backed Blob so a
  // large recording is streamed, not copied into memory; buffer is the
  // fallback for direct callers.
  const blob = filePath
    ? await openAsBlob(filePath, { type: mimetype || 'audio/webm' })
    : new Blob([buffer], { type: mimetype || 'audio/webm' });

  const form = new FormData();
  form.append('file', blob, originalname || 'audio.webm');
  form.append('model', model);

  const res = await fetch(url, {
    method: 'POST',
    headers: { authorization: `Bearer ${process.env.TRANSCRIPTION_API_KEY || ''}` },
    body: form,
  });
  if (!res.ok) {
    throw new Error(
      `ASR provider returned ${res.status}: ${(await res.text()).slice(0, 300)}`,
    );
  }
  const data = await res.json();
  return { text: (data.text ?? '').trim(), mock: false, provider: 'openai-compatible' };
}

// ── NVIDIA NeMo (local, GPU) ────────────────────────────────────────────────
//
// The audio is written to a temp file and handed to a Python sidecar
// (scripts/nemo_transcribe.py) that loads a NeMo ASR model on the GPU and
// prints {"text": ...}. Node stays the orchestrator; the GPU work lives in
// Python, where NeMo actually runs. Exported for direct testing.

export async function transcribeWithNemo(
  { buffer, path: filePath, originalname },
  model,
  { diarize = false } = {},
) {
  const fromResult = (r) => ({
    text: String(r.text ?? '').trim(),
    ...(diarize ? { turns: r.turns ?? null } : {}),
    ...(r.warning ? { warning: String(r.warning) } : {}),
    mock: false,
    provider: 'nemo',
  });
  // Resident worker by default (models stay loaded on the GPU between
  // requests); NEMO_KEEP_WARM=false reverts to the one-shot sidecar.
  const run = (p) =>
    process.env.NEMO_KEEP_WARM === 'false'
      ? runNemo(p, model, diarize)
      : requestNemoWorker(model, { path: p, diarize });

  // Disk-spooled uploads already sit in a temp file (extension preserved by
  // the upload layer) — hand that path straight to the sidecar. The buffer
  // branch serves direct callers that never touched disk.
  if (filePath) {
    return fromResult(await run(filePath));
  }
  const dir = await mkdtemp(path.join(os.tmpdir(), 'nemo-'));
  const inPath = path.join(dir, path.basename(originalname || 'audio.webm'));
  await writeFile(inPath, buffer);
  try {
    return fromResult(await run(inPath));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

// ── Resident NeMo worker ────────────────────────────────────────────────────
//
// One long-lived Python process (scripts/nemo_worker.py) holds the models on
// the GPU; each transcription is a JSON line over stdin/stdout. Spawned
// lazily on first use, respawned on the next request if it dies. Requests
// are matched to replies by id, so a crash rejects exactly the in-flight
// requests and nothing is silently lost.

let nemoWorker = null;

function getNemoWorker(model) {
  if (nemoWorker && nemoWorker.proc.exitCode === null) return nemoWorker;

  const py = process.env.NEMO_PYTHON || 'python3';
  const script = path.join(__dirname, '..', 'scripts', 'nemo_worker.py');
  const proc = spawn(py, [script], { env: { ...process.env, NEMO_MODEL: model } });
  const worker = { proc, pending: new Map(), nextId: 1, buf: '', lastErr: '' };

  proc.stdout.on('data', (d) => {
    worker.buf += d;
    let nl;
    while ((nl = worker.buf.indexOf('\n')) >= 0) {
      const line = worker.buf.slice(0, nl);
      worker.buf = worker.buf.slice(nl + 1);
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        continue; // stray log line on stdout — the reply will still come
      }
      if (!msg || typeof msg !== 'object' || !('id' in msg)) continue;
      const req = worker.pending.get(msg.id);
      if (!req) continue;
      worker.pending.delete(msg.id);
      if (msg.error) req.reject(new Error(`NeMo worker: ${msg.error}`));
      else req.resolve(msg);
    }
  });
  proc.stderr.on('data', (d) => {
    // Rolling tail. Native crash dumps put the informative line well before
    // their ~30 stack frames, so keep enough to include it.
    worker.lastErr = (worker.lastErr + d).slice(-8000);
  });
  const failAll = (err) => {
    for (const req of worker.pending.values()) req.reject(err);
    worker.pending.clear();
    if (nemoWorker === worker) nemoWorker = null;
  };
  proc.on('error', failAll); // e.g. python3 not installed
  proc.on('close', (code) =>
    failAll(new Error(`NeMo worker exited ${code}: ${worker.lastErr.trim().slice(-300)}`)),
  );

  nemoWorker = worker;
  return worker;
}

function requestNemoWorker(model, payload) {
  const worker = getNemoWorker(model);
  const id = worker.nextId++;
  return new Promise((resolve, reject) => {
    worker.pending.set(id, { resolve, reject });
    worker.proc.stdin.write(`${JSON.stringify({ id, ...payload })}\n`);
  });
}

// Preload the ASR model at server startup so even the first dictation is
// fast. Fire-and-forget: a warm-up failure just means the first real
// request pays the load (and surfaces any actual error itself).
export function warmNemo(model) {
  if (process.env.NEMO_KEEP_WARM === 'false') return;
  requestNemoWorker(model, { warm: true }).catch(() => {});
}

// Tests (and graceful shutdowns) can drop the worker; the next request
// respawns it.
export function stopNemoWorker() {
  nemoWorker?.proc.kill();
  nemoWorker = null;
}

function runNemo(inPath, model, diarize = false) {
  return new Promise((resolve, reject) => {
    const py = process.env.NEMO_PYTHON || 'python3';
    const script = path.join(__dirname, '..', 'scripts', 'nemo_transcribe.py');
    const args = [script, inPath, ...(diarize ? ['--diarize'] : [])];
    const proc = spawn(py, args, {
      env: { ...process.env, NEMO_MODEL: model },
    });

    let out = '';
    let err = '';
    proc.stdout.on('data', (d) => (out += d));
    proc.stderr.on('data', (d) => (err += d));
    proc.on('error', reject); // e.g. python3 not installed
    proc.on('close', (code) => {
      if (code !== 0) {
        return reject(
          new Error(`NeMo transcriber exited ${code}: ${err.trim().slice(-300)}`),
        );
      }
      // The result is the last stdout line that is a JSON object with a
      // "text" key — tolerating any stray log lines NeMo may emit on stdout
      // despite the sidecar routing its logging to stderr.
      for (const line of out.trim().split('\n').reverse()) {
        let parsed;
        try {
          parsed = JSON.parse(line);
        } catch {
          continue;
        }
        if (parsed && typeof parsed === 'object' && 'text' in parsed) {
          return resolve(parsed);
        }
      }
      reject(new Error(`NeMo transcriber returned unexpected output: ${out.slice(-200)}`));
    });
  });
}
