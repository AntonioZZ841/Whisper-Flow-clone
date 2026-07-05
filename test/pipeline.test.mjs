// End-to-end tests for the two-stage pipeline.
//
// Spins up a local mock provider (OpenAI-transcription-, DeepSeek-chat- and
// Anthropic-messages-shaped routes) and the real Express app in-process, then
// drives /api/health and /api/transcribe through every provider configuration
// — including an uploaded audio file, asserting the filename/extension is
// forwarded to the ASR provider intact.
//
//   node --test test/

import { test, before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { once } from 'node:events';

import { app } from '../server.js';
import {
  asrConfig,
  resolveAsr,
  transcribe,
  transcribeWithNemo,
  stopNemoWorker,
} from '../src/asr.js';
import { flowConfig, flow } from '../src/flow.js';

const ASR_TEXT = 'um so this is uh a live A S R test no wait an integration test';
const CLEANED = 'Hey, can you move our sync to Tuesday at 3 p.m.?';

let mock;
let mockPort;
let appServer;
let appPort;
let lastAsrFilename = null;

const ENV_KEYS = [
  'TRANSCRIPTION_PROVIDER',
  'TRANSCRIPTION_API_KEY',
  'TRANSCRIPTION_API_URL',
  'TRANSCRIPTION_MODEL',
  'NEMO_MODEL',
  'NEMO_PYTHON',
  'NEMO_ENABLED',
  'NEMO_KEEP_WARM',
  'ANTHROPIC_API_KEY',
  'ANTHROPIC_BASE_URL',
  'ANTHROPIC_AUTH_TOKEN',
  'FLOW_PROVIDER',
  'FLOW_API_KEY',
  'FLOW_API_URL',
  'FLOW_MODEL',
];

// Stub interpreters standing in for `python3 scripts/nemo_transcribe.py` so the
// NeMo Node wiring is testable without a GPU or the nemo_toolkit install.
let stubDir;
let stubOk;
let stubNoisy;
let stubDiar;
let stubFail;
let stubWorker;
let stubWorkerDie;

before(async () => {
  mock = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const body = Buffer.concat(chunks).toString('latin1');
      const send = (obj) => {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify(obj));
      };
      if (req.url.includes('/audio/transcriptions')) {
        lastAsrFilename = (body.match(/filename="([^"]+)"/) || [])[1] || null;
        return send({ text: ASR_TEXT });
      }
      if (req.url.includes('/chat/completions')) {
        return send({ choices: [{ message: { role: 'assistant', content: CLEANED } }] });
      }
      if (req.url.includes('/v1/messages')) {
        return send({
          id: 'msg_mock',
          type: 'message',
          role: 'assistant',
          model: 'claude-haiku-4-5',
          content: [{ type: 'text', text: JSON.stringify({ formatted: CLEANED }) }],
          stop_reason: 'end_turn',
          stop_sequence: null,
          usage: { input_tokens: 10, output_tokens: 10 },
        });
      }
      res.writeHead(404);
      res.end('{}');
    });
  });
  mock.listen(0);
  await once(mock, 'listening');
  mockPort = mock.address().port;

  appServer = app.listen(0);
  await once(appServer, 'listening');
  appPort = appServer.address().port;

  stubDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nemo-stub-'));
  stubOk = path.join(stubDir, 'ok.sh');
  stubNoisy = path.join(stubDir, 'noisy.sh');
  stubDiar = path.join(stubDir, 'diar.sh');
  stubFail = path.join(stubDir, 'fail.sh');
  stubWorker = path.join(stubDir, 'worker.sh');
  stubWorkerDie = path.join(stubDir, 'worker-die.sh');
  // Args ($1=script path, $2=audio path) are ignored — we only exercise the
  // Node-side spawn/parse contract. The noisy stub mimics real NeMo, whose
  // logger writes to stdout before the JSON result line.
  fs.writeFileSync(stubOk, '#!/bin/sh\necho \'{"text":"hey from nemo"}\'\n');
  fs.writeFileSync(
    stubNoisy,
    '#!/bin/sh\necho "[NeMo I 2026-07-05 mixins:184] Tokenizer initialized with 1024 tokens"\necho "loading checkpoint 100%"\necho \'{"text":"hey from noisy nemo"}\'\n',
  );
  fs.writeFileSync(stubFail, '#!/bin/sh\necho "boom" >&2\nexit 1\n');
  // Mimics the sidecar's meeting mode: turns only when --diarize is passed.
  fs.writeFileSync(
    stubDiar,
    '#!/bin/sh\ncase "$*" in\n*--diarize*) echo \'{"text":"hello there general kenobi","turns":[{"speaker":"Speaker 1","start":0,"end":1.1,"text":"hello there"},{"speaker":"Speaker 2","start":1.4,"end":2.6,"text":"general kenobi"}]}\';;\n*) echo \'{"text":"hello there general kenobi"}\';;\nesac\n',
  );
  // A protocol-conforming resident worker: replies per JSON-line request,
  // embedding its PID so tests can prove requests share one process.
  fs.writeFileSync(
    stubWorker,
    `#!/bin/sh
echo '{"ready": true}'
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\\([0-9][0-9]*\\).*/\\1/p')
  case "$line" in
    *'"warm":true'*) printf '{"id": %s, "ok": true}\\n' "$id" ;;
    *'"diarize":true'*) printf '{"id": %s, "text": "warm hello there", "turns": [{"speaker": "Speaker 1", "start": 0, "end": 1, "text": "warm hello"}, {"speaker": "Speaker 2", "start": 1.2, "end": 2, "text": "there"}]}\\n' "$id" ;;
    *) printf '{"id": %s, "text": "warm hello from pid %s"}\\n' "$id" "$$" ;;
  esac
done
`,
  );
  // Reads one request and dies without replying — the crash case.
  fs.writeFileSync(stubWorkerDie, '#!/bin/sh\nread -r line\nexit 7\n');
  fs.chmodSync(stubOk, 0o755);
  fs.chmodSync(stubNoisy, 0o755);
  fs.chmodSync(stubDiar, 0o755);
  fs.chmodSync(stubFail, 0o755);
  fs.chmodSync(stubWorker, 0o755);
  fs.chmodSync(stubWorkerDie, 0o755);
});

after(() => {
  mock?.close();
  appServer?.close();
  stopNemoWorker();
  if (stubDir) fs.rmSync(stubDir, { recursive: true, force: true });
});

beforeEach(() => {
  for (const k of ENV_KEYS) delete process.env[k];
  // Pin auto-resolution off local NeMo so the suite behaves identically on
  // hosts with and without an NVIDIA GPU. The NeMo Node wiring is still
  // covered directly via the NEMO_PYTHON stub interpreters below.
  process.env.NEMO_ENABLED = 'false';
  lastAsrFilename = null;
});

const mockUrl = (p) => `http://127.0.0.1:${mockPort}${p}`;

async function postAudio(filename = 'utterance.webm', type = 'audio/webm', fields = {}) {
  const fd = new FormData();
  fd.append('audio', new Blob([Buffer.from('not-real-audio-mock-asr-ignores-bytes')], { type }), filename);
  for (const [k, v] of Object.entries(fields)) fd.append(k, v);
  const r = await fetch(`http://127.0.0.1:${appPort}/api/transcribe`, { method: 'POST', body: fd });
  return { status: r.status, body: await r.json() };
}

// ── Unit: provider selection ───────────────────────────────────────────────

test('asrConfig: mock with no key/GPU, openai-compatible when a key is set', () => {
  // NEMO_ENABLED=false is pinned in beforeEach, so auto resolves to mock
  // without a key even when the host has an NVIDIA GPU.
  assert.equal(asrConfig().provider, 'mock');
  process.env.TRANSCRIPTION_API_KEY = 'k';
  assert.equal(asrConfig().provider, 'openai-compatible');
  assert.equal(asrConfig().model, 'whisper-1');
});

test('resolveAsr: GPU presence and overrides drive provider selection', () => {
  // auto: GPU present + no cloud key → local NeMo
  assert.deepEqual(resolveAsr({}, true), {
    provider: 'nemo',
    model: 'nvidia/parakeet-tdt-0.6b-v2',
    gpu: true,
  });
  // auto: no GPU, no key → mock
  assert.equal(resolveAsr({}, false).provider, 'mock');
  // auto: a cloud key wins over local NeMo even on a GPU box
  assert.equal(resolveAsr({ TRANSCRIPTION_API_KEY: 'k' }, true).provider, 'openai-compatible');
  // explicit nemo without a GPU degrades gracefully and explains why
  const degraded = resolveAsr({ TRANSCRIPTION_PROVIDER: 'nemo' }, false);
  assert.equal(degraded.provider, 'mock');
  assert.match(degraded.reason, /no NVIDIA GPU/);
  // NEMO_ENABLED=false keeps auto off NeMo on a GPU box
  assert.equal(resolveAsr({ NEMO_ENABLED: 'false' }, true).provider, 'mock');
  // Per-mode routing: with a cloud/Whisper key on a GPU box, plain dictation
  // goes to the key'd provider (multilingual), meetings go to NeMo (the only
  // provider that labels speakers).
  const env = { TRANSCRIPTION_API_KEY: 'k' };
  assert.equal(resolveAsr(env, true).provider, 'openai-compatible');
  assert.equal(resolveAsr(env, true, { diarize: true }).provider, 'nemo');
  // ...but meetings without a GPU fall back to the key'd provider,
  assert.equal(resolveAsr(env, false, { diarize: true }).provider, 'openai-compatible');
  // and an explicit provider override applies to both modes.
  assert.equal(
    resolveAsr({ ...env, TRANSCRIPTION_PROVIDER: 'openai-compatible' }, true, { diarize: true })
      .provider,
    'openai-compatible',
  );
  // custom NeMo model honored
  assert.equal(
    resolveAsr({ TRANSCRIPTION_PROVIDER: 'nemo', NEMO_MODEL: 'nvidia/canary-1b' }, true).model,
    'nvidia/canary-1b',
  );
});

test('flowConfig: auto-selects provider from env', () => {
  assert.equal(flowConfig().provider, 'passthrough');

  process.env.ANTHROPIC_API_KEY = 'k';
  assert.deepEqual(flowConfig(), { provider: 'anthropic', model: 'claude-haiku-4-5' });

  delete process.env.ANTHROPIC_API_KEY;
  process.env.FLOW_API_KEY = 'k';
  assert.deepEqual(flowConfig(), { provider: 'openai-compatible', model: 'deepseek-chat' });

  // Explicit override wins over auto-detection.
  process.env.ANTHROPIC_API_KEY = 'k';
  process.env.FLOW_PROVIDER = 'openai-compatible';
  assert.equal(flowConfig().provider, 'openai-compatible');
});

test('flow: passthrough returns the transcript unchanged', async () => {
  const out = await flow('  hello world  ');
  assert.deepEqual(out, { text: 'hello world', provider: 'passthrough', model: null });
});

test('transcribe: mock mode returns the canned utterance', async () => {
  const out = await transcribe({ buffer: Buffer.from(''), mimetype: 'audio/webm' });
  assert.equal(out.mock, true);
  assert.match(out.text, /no wait actually tuesday/);
});

// ── Integration: /api/health ────────────────────────────────────────────────

test('GET /api/health reflects the active configuration', async () => {
  const r = await fetch(`http://127.0.0.1:${appPort}/api/health`);
  const h = await r.json();
  assert.equal(h.ok, true);
  assert.equal(h.asr.provider, 'mock');
  assert.equal(typeof h.asr.gpu, 'boolean');
  assert.equal(h.flow.provider, 'passthrough');
});

// ── Integration: the three Flow providers ───────────────────────────────────

test('keyless demo mode: mock ASR + passthrough Flow', async () => {
  const { status, body } = await postAudio();
  assert.equal(status, 200);
  assert.match(body.raw, /no wait actually tuesday/);
  assert.equal(body.formatted, body.raw); // passthrough leaves it untouched
  assert.equal(body.meta.flow.provider, 'passthrough');
});

test('openai-compatible (DeepSeek-shaped) Flow cleans the transcript', async () => {
  process.env.TRANSCRIPTION_API_KEY = 'k';
  process.env.TRANSCRIPTION_API_URL = mockUrl('/v1/audio/transcriptions');
  process.env.FLOW_PROVIDER = 'openai-compatible';
  process.env.FLOW_API_KEY = 'k';
  process.env.FLOW_API_URL = mockUrl('/chat/completions');

  const { status, body } = await postAudio();
  assert.equal(status, 200);
  assert.equal(body.raw, ASR_TEXT);
  assert.equal(body.formatted, CLEANED);
  assert.equal(body.meta.flow.provider, 'openai-compatible');
  assert.equal(body.meta.flow.model, 'deepseek-chat');
});

test('anthropic (Claude Haiku 4.5) Flow cleans the transcript', async () => {
  process.env.ANTHROPIC_API_KEY = 'k';
  process.env.ANTHROPIC_BASE_URL = `http://127.0.0.1:${mockPort}`;

  const { status, body } = await postAudio();
  assert.equal(status, 200);
  assert.equal(body.formatted, CLEANED);
  assert.equal(body.meta.flow.provider, 'anthropic');
  assert.equal(body.meta.flow.model, 'claude-haiku-4-5');
});

// ── Integration: uploaded recording ─────────────────────────────────────────

test('uploaded recording is transcribed with its extension preserved', async () => {
  process.env.TRANSCRIPTION_API_KEY = 'k';
  process.env.TRANSCRIPTION_API_URL = mockUrl('/v1/audio/transcriptions');
  // Flow left as passthrough — we're asserting the upload/ASR half here.

  const { status, body } = await postAudio('meeting-recording.wav', 'audio/wav');
  assert.equal(status, 200);
  assert.equal(body.raw, ASR_TEXT);
  // The uploaded filename (and thus format extension) reached the ASR provider.
  assert.equal(lastAsrFilename, 'meeting-recording.wav');
});

test('a large upload (beyond the old 25 MB in-memory cap) is accepted', async () => {
  process.env.TRANSCRIPTION_API_KEY = 'k';
  process.env.TRANSCRIPTION_API_URL = mockUrl('/v1/audio/transcriptions');

  const fd = new FormData();
  fd.append(
    'audio',
    new Blob([Buffer.alloc(30 * 1024 * 1024)], { type: 'audio/wav' }),
    'long-meeting.wav',
  );
  const r = await fetch(`http://127.0.0.1:${appPort}/api/transcribe`, { method: 'POST', body: fd });
  assert.equal(r.status, 200);
  assert.equal((await r.json()).raw, ASR_TEXT);
  assert.equal(lastAsrFilename, 'long-meeting.wav');
});

test('flow stage falls back to raw transcript when the provider errors', async () => {
  process.env.FLOW_PROVIDER = 'openai-compatible';
  process.env.FLOW_API_KEY = 'k';
  process.env.FLOW_API_URL = mockUrl('/nonexistent-route-yields-404');

  const out = await flow('um hello there');
  assert.equal(out.text, 'um hello there'); // never loses the dictation
  assert.equal(out.provider, 'passthrough');
  assert.ok(out.error);
});

// ── NeMo provider (Node-side wiring, GPU/model stubbed) ─────────────────────

test('transcribeWithNemo: writes a temp file, runs the sidecar, parses JSON', async () => {
  process.env.NEMO_KEEP_WARM = 'false';
  process.env.NEMO_PYTHON = stubOk;
  const out = await transcribeWithNemo(
    { buffer: Buffer.from('audio'), originalname: 'clip.webm' },
    'nvidia/parakeet-tdt-0.6b-v2',
  );
  assert.deepEqual(out, { text: 'hey from nemo', mock: false, provider: 'nemo' });
});

test('transcribeWithNemo: --diarize is forwarded and speaker turns are parsed', async () => {
  process.env.NEMO_KEEP_WARM = 'false'; // exercise the one-shot sidecar contract
  process.env.NEMO_PYTHON = stubDiar;
  const plain = await transcribeWithNemo(
    { buffer: Buffer.from('audio'), originalname: 'meeting.wav' },
    'm',
  );
  assert.equal(plain.text, 'hello there general kenobi');
  assert.equal('turns' in plain, false); // no meeting mode → no turns key

  const meeting = await transcribeWithNemo(
    { buffer: Buffer.from('audio'), originalname: 'meeting.wav' },
    'm',
    { diarize: true },
  );
  assert.equal(meeting.turns.length, 2);
  assert.deepEqual(meeting.turns[1], {
    speaker: 'Speaker 2',
    start: 1.4,
    end: 2.6,
    text: 'general kenobi',
  });
});

test('keyless meeting mode: mock ASR returns labeled speaker turns', async () => {
  const { status, body } = await postAudio('standup.wav', 'audio/wav', { diarize: 'true' });
  assert.equal(status, 200);
  assert.ok(body.turns.length >= 2);
  assert.equal(body.turns[0].speaker, 'Speaker 1');
  assert.match(body.raw, /^Speaker 1: /);
  assert.match(body.raw, /\nSpeaker 2: /);
  assert.equal(body.formatted, body.raw); // passthrough Flow leaves it as-is
});

test('transcribeWithNemo: tolerates NeMo log noise on stdout before the JSON', async () => {
  process.env.NEMO_KEEP_WARM = 'false';
  process.env.NEMO_PYTHON = stubNoisy;
  const out = await transcribeWithNemo(
    { buffer: Buffer.from('audio'), originalname: 'clip.webm' },
    'nvidia/parakeet-tdt-0.6b-v2',
  );
  assert.deepEqual(out, { text: 'hey from noisy nemo', mock: false, provider: 'nemo' });
});

test('transcribeWithNemo: surfaces a non-zero exit from the sidecar', async () => {
  process.env.NEMO_KEEP_WARM = 'false';
  process.env.NEMO_PYTHON = stubFail;
  await assert.rejects(
    () => transcribeWithNemo({ buffer: Buffer.from('x'), originalname: 'c.webm' }, 'm'),
    /NeMo transcriber exited 1.*boom/s,
  );
});

// ── Resident NeMo worker (default path: models stay loaded) ────────────────

test('nemo worker: consecutive requests reuse one resident process', async () => {
  process.env.NEMO_PYTHON = stubWorker;
  stopNemoWorker(); // fresh worker for this test
  const file = { buffer: Buffer.from('audio'), originalname: 'a.webm' };

  const first = await transcribeWithNemo(file, 'm');
  const second = await transcribeWithNemo(file, 'm');
  assert.match(first.text, /^warm hello from pid \d+$/);
  assert.equal(second.text, first.text); // same PID → same process, no reload

  stopNemoWorker();
  const third = await transcribeWithNemo(file, 'm');
  assert.notEqual(third.text, first.text); // stopped → respawned
});

test('nemo worker: speaker turns come back through the resident protocol', async () => {
  process.env.NEMO_PYTHON = stubWorker;
  stopNemoWorker();
  const out = await transcribeWithNemo(
    { buffer: Buffer.from('audio'), originalname: 'meeting.wav' },
    'm',
    { diarize: true },
  );
  assert.equal(out.turns.length, 2);
  assert.equal(out.turns[0].speaker, 'Speaker 1');
});

test('nemo worker: a dying worker rejects the request, then respawns clean', async () => {
  process.env.NEMO_PYTHON = stubWorkerDie;
  stopNemoWorker();
  const file = { buffer: Buffer.from('audio'), originalname: 'a.webm' };
  await assert.rejects(() => transcribeWithNemo(file, 'm'), /NeMo worker exited 7/);

  process.env.NEMO_PYTHON = stubWorker;
  const out = await transcribeWithNemo(file, 'm');
  assert.match(out.text, /^warm hello from pid \d+$/);
});

// ── Bad request ─────────────────────────────────────────────────────────────

test('POST /api/transcribe with no file returns 400', async () => {
  const r = await fetch(`http://127.0.0.1:${appPort}/api/transcribe`, { method: 'POST' });
  assert.equal(r.status, 400);
});
