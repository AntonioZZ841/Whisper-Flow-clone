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
import { once } from 'node:events';

import { app } from '../server.js';
import { asrConfig, transcribe } from '../src/asr.js';
import { flowConfig, flow } from '../src/flow.js';

const ASR_TEXT = 'um so this is uh a live A S R test no wait an integration test';
const CLEANED = 'Hey, can you move our sync to Tuesday at 3 p.m.?';

let mock;
let mockPort;
let appServer;
let appPort;
let lastAsrFilename = null;

const ENV_KEYS = [
  'TRANSCRIPTION_API_KEY',
  'TRANSCRIPTION_API_URL',
  'TRANSCRIPTION_MODEL',
  'ANTHROPIC_API_KEY',
  'ANTHROPIC_BASE_URL',
  'ANTHROPIC_AUTH_TOKEN',
  'FLOW_PROVIDER',
  'FLOW_API_KEY',
  'FLOW_API_URL',
  'FLOW_MODEL',
];

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
          model: 'claude-sonnet-5',
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
});

after(() => {
  mock?.close();
  appServer?.close();
});

beforeEach(() => {
  for (const k of ENV_KEYS) delete process.env[k];
  lastAsrFilename = null;
});

const mockUrl = (p) => `http://127.0.0.1:${mockPort}${p}`;

async function postAudio(filename = 'utterance.webm', type = 'audio/webm') {
  const fd = new FormData();
  fd.append('audio', new Blob([Buffer.from('not-real-audio-mock-asr-ignores-bytes')], { type }), filename);
  const r = await fetch(`http://127.0.0.1:${appPort}/api/transcribe`, { method: 'POST', body: fd });
  return { status: r.status, body: await r.json() };
}

// ── Unit: provider selection ───────────────────────────────────────────────

test('asrConfig: mock when no key, live when key set', () => {
  assert.equal(asrConfig().mode, 'mock');
  process.env.TRANSCRIPTION_API_KEY = 'k';
  assert.equal(asrConfig().mode, 'live');
  assert.equal(asrConfig().model, 'whisper-1');
});

test('flowConfig: auto-selects provider from env', () => {
  assert.equal(flowConfig().provider, 'passthrough');

  process.env.ANTHROPIC_API_KEY = 'k';
  assert.deepEqual(flowConfig(), { provider: 'anthropic', model: 'claude-sonnet-5' });

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
  assert.equal(h.asr.mode, 'mock');
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

test('anthropic (Claude Sonnet 5) Flow cleans the transcript', async () => {
  process.env.ANTHROPIC_API_KEY = 'k';
  process.env.ANTHROPIC_BASE_URL = `http://127.0.0.1:${mockPort}`;

  const { status, body } = await postAudio();
  assert.equal(status, 200);
  assert.equal(body.formatted, CLEANED);
  assert.equal(body.meta.flow.provider, 'anthropic');
  assert.equal(body.meta.flow.model, 'claude-sonnet-5');
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

test('flow stage falls back to raw transcript when the provider errors', async () => {
  process.env.FLOW_PROVIDER = 'openai-compatible';
  process.env.FLOW_API_KEY = 'k';
  process.env.FLOW_API_URL = mockUrl('/nonexistent-route-yields-404');

  const out = await flow('um hello there');
  assert.equal(out.text, 'um hello there'); // never loses the dictation
  assert.equal(out.provider, 'passthrough');
  assert.ok(out.error);
});

// ── Bad request ─────────────────────────────────────────────────────────────

test('POST /api/transcribe with no file returns 400', async () => {
  const r = await fetch(`http://127.0.0.1:${appPort}/api/transcribe`, { method: 'POST' });
  assert.equal(r.status, 400);
});
