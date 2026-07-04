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
| Stage 1 ASR | Cloud Whisper-family + in-house | Any OpenAI-compatible Whisper endpoint (or mock) |
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

- **`src/asr.js`** — forwards the audio blob to an OpenAI-compatible
  `/audio/transcriptions` endpoint. No key → mock mode (canned disfluent
  utterance so the pipeline is demoable with zero credentials).
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
| Real transcription | `TRANSCRIPTION_API_KEY` (OpenAI, or point `TRANSCRIPTION_API_URL` at your own whisper.cpp/faster-whisper server) |
| Cleanup with Claude Sonnet 5 | `ANTHROPIC_API_KEY` |
| Cleanup with DeepSeek (or any OpenAI-compatible LLM) | `FLOW_PROVIDER=openai-compatible`, `FLOW_API_KEY`, and optionally `FLOW_API_URL` / `FLOW_MODEL` |

> Microphone capture requires a secure context: `localhost` is fine, remote
> hosts need HTTPS.

## API

- `POST /api/transcribe` — multipart form with an `audio` file. Returns
  `{ raw, formatted, meta }` where `meta` reports which providers ran.
- `GET /api/health` — `{ ok, asr: {mode, model}, flow: {provider, model} }`.

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
