// Stage 1 — ASR (speech → raw transcript).
//
// Wispr Flow transcribes in the cloud with Whisper-family models and
// processes one utterance at a time ("wait, understand, then write what you
// meant") rather than streaming word-by-word. This module does the same
// against any OpenAI-compatible /audio/transcriptions endpoint —
// api.openai.com by default, or a self-hosted whisper.cpp / faster-whisper
// server via TRANSCRIPTION_API_URL.
//
// With no TRANSCRIPTION_API_KEY set it runs in mock mode and returns a
// canned disfluent utterance, so the UI and the Flow stage can be exercised
// without any credentials.

const MOCK_TRANSCRIPT =
  'um so hey can you move our sync to monday no wait actually tuesday at uh 3 p.m. and um also like send the notes uh send the notes to everyone after';

export function asrConfig() {
  const live = Boolean(process.env.TRANSCRIPTION_API_KEY);
  return {
    mode: live ? 'live' : 'mock',
    model: live ? process.env.TRANSCRIPTION_MODEL || 'whisper-1' : null,
  };
}

export async function transcribe({ buffer, mimetype, originalname }) {
  const key = process.env.TRANSCRIPTION_API_KEY;
  if (!key) {
    return { text: MOCK_TRANSCRIPT, mock: true };
  }

  const url =
    process.env.TRANSCRIPTION_API_URL ||
    'https://api.openai.com/v1/audio/transcriptions';

  const form = new FormData();
  form.append(
    'file',
    new Blob([buffer], { type: mimetype || 'audio/webm' }),
    originalname || 'audio.webm',
  );
  form.append('model', process.env.TRANSCRIPTION_MODEL || 'whisper-1');

  const res = await fetch(url, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}` },
    body: form,
  });
  if (!res.ok) {
    throw new Error(
      `ASR provider returned ${res.status}: ${(await res.text()).slice(0, 300)}`,
    );
  }
  const data = await res.json();
  return { text: (data.text ?? '').trim(), mock: false };
}
