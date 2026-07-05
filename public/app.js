// Whisper Flow clone — browser side.
//
// Captures one utterance (push-to-talk), sends it through the two-stage
// pipeline, and inserts the formatted result at the cursor — the closest a
// web page gets to Wispr Flow's "type where you are" behavior.

const els = {
  ptt: document.getElementById('ptt'),
  pttLabel: document.getElementById('ptt-label'),
  status: document.getElementById('status'),
  providers: document.getElementById('providers'),
  timer: document.getElementById('timer'),
  meter: document.getElementById('meter'),
  formatted: document.getElementById('formatted'),
  raw: document.getElementById('raw'),
  copy: document.getElementById('copy'),
  download: document.getElementById('download'),
  clear: document.getElementById('clear'),
  words: document.getElementById('word-count'),
  file: document.getElementById('file'),
  drop: document.getElementById('drop'),
  fileName: document.getElementById('file-name'),
  diarize: document.getElementById('diarize'),
};

// Presses shorter than this toggle hands-free mode; longer ones behave like
// a walkie-talkie (record while held, stop on release) — Wispr's two
// activation modes, minus the OS-global hotkey a browser can't have.
const HOLD_THRESHOLD_MS = 350;

const state = {
  recorder: null,
  stream: null,
  chunks: [],
  recording: false,
  busy: false,
  pressStartedAt: 0,
  pressStartedRecording: false,
  audioCtx: null,
  raf: 0,
  timerId: 0,
  startedAt: 0,
};

init();

function init() {
  loadHealth();

  els.ptt.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    pressStart();
  });
  els.ptt.addEventListener('pointerup', pressEnd);
  els.ptt.addEventListener('pointercancel', pressEnd);
  els.ptt.addEventListener('pointerleave', pressEnd);

  // Space bar as an in-page stand-in for Wispr's global Fn hotkey.
  document.addEventListener('keydown', (e) => {
    if (e.code !== 'Space' || e.repeat || isEditable(e.target)) return;
    e.preventDefault();
    pressStart();
  });
  document.addEventListener('keyup', (e) => {
    if (e.code !== 'Space' || isEditable(e.target)) return;
    e.preventDefault();
    pressEnd();
  });

  els.copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(els.formatted.value);
      flash(els.copy, 'Copied!');
    } catch {
      els.formatted.select();
      document.execCommand('copy');
      flash(els.copy, 'Copied!');
    }
  });

  els.download.addEventListener('click', () => {
    const blob = new Blob([els.formatted.value], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'dictation.txt';
    a.click();
    URL.revokeObjectURL(a.href);
  });

  els.clear.addEventListener('click', () => {
    els.formatted.value = '';
    els.raw.textContent = '';
    refreshOutputs();
  });

  els.formatted.addEventListener('input', refreshOutputs);

  // Upload an existing recording instead of speaking live — same pipeline.
  els.file.addEventListener('change', () => {
    if (els.file.files[0]) uploadFile(els.file.files[0]);
    els.file.value = ''; // allow re-selecting the same file
  });

  for (const evt of ['dragenter', 'dragover']) {
    els.drop.addEventListener(evt, (e) => {
      e.preventDefault();
      els.drop.classList.add('upload__drop--over');
    });
  }
  for (const evt of ['dragleave', 'drop']) {
    els.drop.addEventListener(evt, (e) => {
      e.preventDefault();
      els.drop.classList.remove('upload__drop--over');
    });
  }
  els.drop.addEventListener('drop', (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) uploadFile(f);
  });
}

function uploadFile(file) {
  if (state.busy || state.recording) return;
  if (!/^audio\//.test(file.type) && !/\.(webm|m4a|mp3|wav|ogg|flac|aac|opus)$/i.test(file.name)) {
    setStatus('error', 'not an audio file');
    return;
  }
  els.fileName.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  send(file, file.name, { diarize: els.diarize.checked });
}

function isEditable(t) {
  return (
    t instanceof HTMLElement &&
    (t.tagName === 'TEXTAREA' ||
      t.tagName === 'INPUT' ||
      t.tagName === 'SELECT' ||
      t.isContentEditable)
  );
}

async function loadHealth() {
  try {
    const res = await fetch('/api/health');
    const h = await res.json();
    const asr =
      h.asr.provider === 'nemo'
        ? `ASR NeMo (${h.asr.model})`
        : h.asr.provider === 'openai-compatible'
          ? `ASR ${h.asr.model}`
          : 'ASR mock';
    const flow =
      h.flow.provider === 'passthrough'
        ? 'Flow passthrough'
        : `Flow ${h.flow.model}`;
    const gpu = h.asr.gpu ? ' · GPU' : '';
    els.providers.textContent = `${asr} · ${flow}${gpu}`;
  } catch {
    els.providers.textContent = 'server unreachable';
  }
}

// ── Push-to-talk press logic ────────────────────────────────────────────

function pressStart() {
  if (state.busy) return;
  if (!state.recording) {
    state.pressStartedAt = Date.now();
    state.pressStartedRecording = true;
    startRecording();
  } else {
    // Second tap while hands-free → stop.
    state.pressStartedRecording = false;
    stopRecording();
  }
}

function pressEnd() {
  const wasHold =
    state.recording &&
    state.pressStartedRecording &&
    Date.now() - state.pressStartedAt >= HOLD_THRESHOLD_MS;
  state.pressStartedRecording = false;
  if (wasHold) stopRecording();
}

// ── Recording lifecycle ─────────────────────────────────────────────────

async function startRecording() {
  if (state.recording) return;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setStatus('error', 'this browser has no microphone API');
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    setStatus('error', 'microphone permission denied');
    return;
  }

  state.chunks = [];
  state.recorder = new MediaRecorder(state.stream);
  state.recorder.ondataavailable = (e) => {
    if (e.data.size) state.chunks.push(e.data);
  };
  state.recorder.onstop = onRecorderStop;
  state.recorder.start();

  state.recording = true;
  state.startedAt = Date.now();
  els.ptt.classList.add('recording');
  els.pttLabel.textContent = 'Listening…';
  els.timer.hidden = false;
  els.meter.hidden = false;
  updateTimer();
  state.timerId = setInterval(updateTimer, 250);
  startMeter();
  setStatus('recording', 'recording');
}

function stopRecording() {
  if (!state.recording) return;
  state.recording = false;
  clearInterval(state.timerId);
  stopMeter();
  els.ptt.classList.remove('recording');
  els.pttLabel.textContent = 'Hold to talk';
  els.timer.hidden = true;
  els.meter.hidden = true;
  state.recorder.stop(); // fires onRecorderStop with the collected chunks
}

function onRecorderStop() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  const blob = new Blob(state.chunks, {
    type: state.recorder?.mimeType || 'audio/webm',
  });
  if (blob.size === 0) {
    setStatus('idle', 'ready');
    return;
  }
  // The "Label speakers" toggle applies to mic recordings too — a phone on
  // the table capturing a live meeting is exactly the meeting-mode use case.
  send(blob, 'utterance.webm', { diarize: els.diarize.checked });
}

function updateTimer() {
  const s = Math.floor((Date.now() - state.startedAt) / 1000);
  els.timer.textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

// ── Level meter ─────────────────────────────────────────────────────────

function startMeter() {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return;
  state.audioCtx = new AudioCtx();
  const source = state.audioCtx.createMediaStreamSource(state.stream);
  const analyser = state.audioCtx.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);

  const data = new Uint8Array(analyser.frequencyBinCount);
  const ctx = els.meter.getContext('2d');
  const accent =
    getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() ||
    '#e4584a';

  const draw = () => {
    analyser.getByteFrequencyData(data);
    const { width, height } = els.meter;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = accent;
    const bars = 48;
    const step = Math.floor(data.length / bars);
    const w = width / bars;
    for (let i = 0; i < bars; i++) {
      const v = data[i * step] / 255;
      const h = Math.max(2, v * height);
      ctx.fillRect(i * w + w * 0.2, (height - h) / 2, w * 0.6, h);
    }
    state.raf = requestAnimationFrame(draw);
  };
  draw();
}

function stopMeter() {
  cancelAnimationFrame(state.raf);
  state.audioCtx?.close();
  state.audioCtx = null;
  els.meter.getContext('2d').clearRect(0, 0, els.meter.width, els.meter.height);
}

// ── Pipeline round-trip ─────────────────────────────────────────────────

async function send(blob, filename = 'utterance.webm', { diarize = false } = {}) {
  state.busy = true;
  setStatus('busy', diarize ? 'transcribing + labeling speakers…' : 'transcribing + formatting…');

  const fd = new FormData();
  // Preserve the filename — Whisper-style ASR uses the extension to detect
  // the audio format, so an uploaded recording.wav must arrive as .wav.
  fd.append('audio', blob, filename);
  if (diarize) fd.append('diarize', 'true');

  try {
    const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `server returned ${res.status}`);

    els.raw.textContent = data.raw || '(silence)';
    insertAtCursor(data.formatted || '');
    refreshOutputs();
    const warning = data.meta?.asr?.warning;
    if (warning) setStatus('idle', `done — ${warning}`);
    else setStatus('idle', data.meta?.asr?.mock ? 'done (mock ASR)' : 'done');
  } catch (err) {
    setStatus('error', String(err.message || err));
  } finally {
    state.busy = false;
  }
}

// Insert at the caret like Wispr types at your cursor, instead of
// replacing the whole document.
function insertAtCursor(text) {
  if (!text) return;
  const ta = els.formatted;
  const before = ta.value.slice(0, ta.selectionStart);
  // Multi-line insertions (labeled meeting transcripts) get their own block.
  const sep = before && !/\s$/.test(before) ? (text.includes('\n') ? '\n\n' : ' ') : '';
  ta.setRangeText(sep + text, ta.selectionStart, ta.selectionEnd, 'end');
}

// ── UI helpers ──────────────────────────────────────────────────────────

function refreshOutputs() {
  const words = els.formatted.value.trim().split(/\s+/).filter(Boolean).length;
  els.words.textContent = `${words} word${words === 1 ? '' : 's'}`;
  const empty = els.formatted.value.trim() === '' && els.raw.textContent === '';
  for (const btn of [els.copy, els.download, els.clear]) btn.disabled = empty;
}

function setStatus(kind, text) {
  els.status.className = `pill pill--${kind}`;
  els.status.textContent = text;
}

function flash(btn, text) {
  const original = btn.textContent;
  btn.textContent = text;
  setTimeout(() => (btn.textContent = original), 1200);
}
