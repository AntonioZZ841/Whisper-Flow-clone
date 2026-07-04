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
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const MOCK_TRANSCRIPT =
  'um so hey can you move our sync to monday no wait actually tuesday at uh 3 p.m. and um also like send the notes uh send the notes to everyone after';

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

export async function transcribe(file) {
  const cfg = asrConfig();
  if (cfg.provider === 'mock') {
    return { text: MOCK_TRANSCRIPT, mock: true, provider: 'mock' };
  }
  if (cfg.provider === 'nemo') {
    return transcribeWithNemo(file, cfg.model);
  }
  return transcribeOpenAICompatible(file, cfg.model);
}

// ── openai-compatible (OpenAI / whisper.cpp / faster-whisper) ───────────────

async function transcribeOpenAICompatible({ buffer, mimetype, originalname }, model) {
  const url =
    process.env.TRANSCRIPTION_API_URL ||
    'https://api.openai.com/v1/audio/transcriptions';

  const form = new FormData();
  form.append(
    'file',
    new Blob([buffer], { type: mimetype || 'audio/webm' }),
    originalname || 'audio.webm',
  );
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

export async function transcribeWithNemo({ buffer, originalname }, model) {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'nemo-'));
  const inPath = path.join(dir, path.basename(originalname || 'audio.webm'));
  await writeFile(inPath, buffer);
  try {
    const text = await runNemo(inPath, model);
    return { text: text.trim(), mock: false, provider: 'nemo' };
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

function runNemo(inPath, model) {
  return new Promise((resolve, reject) => {
    const py = process.env.NEMO_PYTHON || 'python3';
    const script = path.join(__dirname, '..', 'scripts', 'nemo_transcribe.py');
    const proc = spawn(py, [script, inPath], {
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
      try {
        resolve(String(JSON.parse(out).text ?? ''));
      } catch {
        reject(new Error(`NeMo transcriber returned unexpected output: ${out.slice(0, 200)}`));
      }
    });
  });
}
