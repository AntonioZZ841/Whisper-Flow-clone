# 🎙️ Whisper Flow clone

An open-source, browser-based clone of [Wispr Flow](https://wisprflow.ai) — the
voice-dictation tool that doesn't just transcribe what you *said*, but writes
what you *meant to type*.

Speak with all your "um"s, false starts, and mid-sentence corrections; get back
clean, punctuated text with the corrections resolved.

```
you say:   "um so hey can you move our sync to monday no wait actually
            tuesday at uh 3 p.m."

you get:   "Hey, can you move our sync to Tuesday at 3 p.m.?"
```

---

## How the real thing works (mechanism deep-dive)

Wispr Flow's defining insight is that **raw speech-to-text output is an
intermediate representation, not the product**. Their pipeline, as documented
in their engineering material:

```
hold global hotkey → mic audio → cloud
    Stage 1  ASR            Whisper-family / in-house models, selected
                            per detected language
    Stage 2  LLM cleanup    fine-tuned Llama (served on Baseten, <700ms p99)
→ finished text auto-typed into whatever app has focus
```

Key mechanisms of the real product:

- **Per-utterance processing, not token streaming.** Their stated philosophy is
  "wait, understand, then write what you meant" — the model needs the whole
  utterance as context before it can clean it correctly.
- **Smart Formatting** — the LLM stage removes fillers and disfluencies, adds
  punctuation and capitalization, and fixes grammar.
- **Backtrack** — spoken self-corrections ("no wait", "scratch that", plain
  restatements) are resolved to the final version, using the full utterance as
  context; ambiguous cases ("I actually enjoyed it") are left alone.
- **Context awareness** *(native-only)* — reads the active app, text near the
  cursor, and on-screen proper nouns via OS accessibility APIs to bias
  transcription and pick a per-app writing style.
- **Auto-insertion** *(native-only)* — types the result into any focused field
  through macOS Accessibility / Windows automation APIs.
- **Personal dictionary** — learns names and jargon from your corrections.

Sources: [Baseten × Wispr Flow case study](https://www.baseten.co/resources/customers/wispr-flow/),
[Wispr on language support](https://wisprflow.ai/research/supporting-languages),
[Wispr on voice-interface design](https://wisprflow.ai/post/designing-a-natural-and-useful-voice-interface),
[Wispr docs](https://docs.wisprflow.ai).

## What this clone implements

| Mechanism | Real Wispr Flow | This clone |
| --- | --- | --- |
| Push-to-talk capture | Global Fn hotkey, any app | In-page button / space bar (hold or tap-toggle) |
| Transcribe an existing recording | — | Upload / drag-and-drop an audio file (webm, mp3, wav, m4a, ogg, flac) |
| Stage 1 ASR | Cloud Whisper-family + in-house | Any OpenAI-compatible Whisper endpoint, **local NVIDIA NeMo** (GPU), or mock |
| Stage 2 cleanup | Fine-tuned Llama on Baseten | **Claude Sonnet 5** by default, or any OpenAI-compatible chat API (e.g. **DeepSeek**) |
| Smart Formatting + Backtrack | ✅ | ✅ (the Flow stage's system prompt) |
| Per-utterance pipeline | ✅ | ✅ |
| Auto-typing into any app | ✅ (accessibility APIs) | ❌ browser sandbox — inserts at the cursor of the in-page editor instead |
| Context awareness / dictionary / command mode | ✅ | ❌ out of scope |

## Architecture

```
browser (public/)                      server (Node + Express)
┌───────────────────────┐   audio    ┌──────────────────────────────┐
│ MediaRecorder capture │──────────▶│ POST /api/transcribe          │
│ push-to-talk UI       │            │  ├─ src/asr.js   Stage 1 ASR │
│ raw vs formatted view │◀──────────│  └─ src/flow.js  Stage 2 Flow│
└───────────────────────┘ raw+clean  └──────────────────────────────┘
```

- **`src/asr.js`** — resolves the ASR provider (`resolveAsr`, unit-tested) and
  transcribes:
  - **`openai-compatible`** — forwards the audio to a `/audio/transcriptions`
    endpoint (OpenAI, whisper.cpp, faster-whisper).
  - **`nemo`** — local **NVIDIA NeMo** ASR, offered only when the server has an
    NVIDIA GPU (detected via `nvidia-smi`). Node writes the clip to a temp file
    and runs the [`scripts/nemo_transcribe.py`](scripts/nemo_transcribe.py)
    sidecar, which loads a NeMo model (Parakeet/Canary) on the GPU.
  - **`mock`** — no key and no GPU: a canned disfluent utterance so the pipeline
    is demoable with zero credentials.
- **`src/flow.js`** — the Flow stage. Sends the raw transcript plus a
  Smart-Formatting-and-Backtrack system prompt to the configured LLM:
  - **`anthropic`** *(default)* — `claude-sonnet-5` via the official SDK, with
    thinking disabled and `effort: low` (dictation must feel instant) and a
    JSON-schema structured output.
  - **`openai-compatible`** — any `/chat/completions` server; the defaults
    target DeepSeek (`deepseek-chat`). Works the same with a local llama.cpp,
    Ollama, vLLM, etc.
  - **`passthrough`** — no key: the raw transcript is returned unchanged.
  On any provider error the stage falls back to the raw transcript — cleanup
  must never lose a dictation.
- **`server.js`** — wires the two stages and serves the static UI.

## Quickstart

```bash
npm install
cp .env.example .env   # optional — runs in demo mode without it
npm start              # → http://localhost:3000
```

With no keys configured you get **demo mode**: the ASR stage returns a canned
disfluent utterance and the Flow stage passes it through, so you can exercise
the whole UI. Add keys to light up each stage independently:

| You want | Set |
| --- | --- |
| Cloud transcription | `TRANSCRIPTION_API_KEY` (OpenAI, or point `TRANSCRIPTION_API_URL` at your own whisper.cpp/faster-whisper server) |
| Local transcription on a GPU box | `TRANSCRIPTION_PROVIDER=nemo` (see below) |
| Cleanup with Claude Sonnet 5 | `ANTHROPIC_API_KEY` |
| Cleanup with DeepSeek (or any OpenAI-compatible LLM) | `FLOW_PROVIDER=openai-compatible`, `FLOW_API_KEY`, and optionally `FLOW_API_URL` / `FLOW_MODEL` |

Provider selection defaults to **`auto`**: a cloud key wins if set, otherwise
local NeMo when an NVIDIA GPU is present, otherwise mock.

> Microphone capture requires a secure context: `localhost` is fine, remote
> hosts need HTTPS.

### HTTPS for LAN/mobile testing

To try the app from another device on your network (a phone, say) instead of
just `localhost`, generate a self-signed dev certificate once:

```bash
npm run cert    # writes certs/dev-{key,cert}.pem, covering localhost + your LAN IPs
npm start       # detects the cert automatically and switches to HTTPS
```

The server prints the LAN URL(s) to open on other devices. Browsers will warn
on first visit since the certificate is self-signed — accept the risk once
per device to continue. This is a dev convenience only; a public deployment
should sit behind a reverse proxy with a real certificate (e.g. Caddy or
nginx with Let's Encrypt) instead of this self-signed one.

The server already listens on every network interface by default (`HOST=0.0.0.0`,
overridable in `.env`) — Node's default, not something you need to change. If
a LAN device still can't connect, that's almost always the OS firewall (e.g.
Windows Defender Firewall on a WSL2 host) blocking the inbound port rather
than the app.

### Local transcription with NVIDIA NeMo (GPU)

If the deployment server has an NVIDIA GPU, you can transcribe entirely
on-device with [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — no cloud ASR key.
The `nemo` provider is offered only when a GPU is detected (`nvidia-smi`);
requesting it without one degrades gracefully to mock/cloud and says why.

Prerequisites on the server:

```bash
pip install "nemo_toolkit[asr]"   # pulls in torch; use a CUDA-enabled build
# ffmpeg must be on PATH (browser clips are webm/opus → 16 kHz mono WAV)
```

Then:

```bash
TRANSCRIPTION_PROVIDER=nemo NEMO_MODEL=nvidia/parakeet-tdt-0.6b-v2 npm start
```

The first request downloads and loads the model; subsequent ones reuse the
cache. For a busy deployment, run a resident NeMo service that exposes an
OpenAI-compatible `/audio/transcriptions` endpoint and use the
`openai-compatible` provider pointed at it, so the model stays warm between
requests (`scripts/nemo_transcribe.py` documents the tradeoff).

### Meeting transcription — who said what (speaker diarization)

Tick **“Label speakers”** next to the upload zone (or POST with a
`diarize=true` form field) and the transcript comes back segmented by voice:

```
Speaker 1: Are we still on for the quarterly review on Thursday?
Speaker 2: Yes, Thursday works for me. Can you send the invite?
```

How it works: the ASR pass also emits word timestamps, NVIDIA **Sortformer**
(`nvidia/diar_sortformer_4spk-v1`, up to 4 speakers) segments the audio by
voice on the same GPU, and the sidecar merges the two — each word goes to the
speaker whose segment covers it. The Flow stage then cleans each turn while
preserving the labels. The response carries both the labeled text (`raw`,
`formatted`) and structured `turns` (`{speaker, start, end, text}`).

Notes:

- Needs the local **NeMo provider** (GPU). The cloud `openai-compatible`
  provider has no diarization concept — you get an unlabeled transcript plus
  a warning. Keyless mock mode returns a canned two-person exchange so the
  UI can be exercised without a GPU.
- Speakers are labeled by order of first appearance (`Speaker 1`, `Speaker
  2`, …) — telling voices apart, not recognizing *whose* voice; no voice
  profiles are stored.
- The first meeting request downloads the diarization model (~700 MB);
  override with `NEMO_DIAR_MODEL`. If diarization fails, the transcript is
  still returned unlabeled with a warning — labeling degrades, dictation is
  never lost.

## Two ways to get audio in

- **Speak live** — hold the button (or the space bar) and talk; release to
  finish. A quick tap toggles hands-free mode.
- **Upload a recording** — drag an audio file onto the drop zone, or click to
  pick one (`webm`, `mp3`, `wav`, `m4a`, `ogg`, `flac`). The original filename
  is forwarded so the ASR engine detects the format from its extension.
  Uploads spool to a temp file rather than RAM, so long recordings (up to
  `MAX_UPLOAD_MB`, default 200 MB) are fine; oversized uploads get a clean 413.

Both go through the exact same `ASR → Flow` pipeline.

## API

- `POST /api/transcribe` — multipart form with an `audio` file (a live-recorded
  blob or an uploaded file), plus optional `diarize=true` for meeting mode.
  Returns `{ raw, formatted, meta }` — with `diarize`, also `turns:
  [{speaker, start, end, text}] | null` and speaker-labeled `raw`/`formatted`.
- `GET /api/health` — `{ ok, asr: {mode, model}, flow: {provider, model} }`.

## Testing

A committed [`node:test`](test/pipeline.test.mjs) suite spins up the Express app
with a local mock provider and exercises config selection, all three Flow
providers, uploaded-file transcription (asserting the extension is forwarded),
and the raw-transcript fallback:

```bash
npm test
```

## What a browser clone can't do

The pieces of Wispr Flow that require a native app — a **global** hotkey that
works while other apps have focus, **auto-typing into any application**
(macOS Accessibility / Windows UI Automation), and **screen/context
awareness** — are impossible in a browser sandbox by design. The path to
adding them is wrapping this same pipeline in an Electron or Tauri shell,
which is exactly how open-source native clones like
[OpenWhispr](https://github.com/OpenWhispr/openwhispr) are built.

## License

MIT
