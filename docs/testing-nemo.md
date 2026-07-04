# Testing the NVIDIA NeMo ASR path

The `nemo` transcription provider runs [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
ASR locally and is only offered when the server has an NVIDIA GPU (detected via
`nvidia-smi`). This guide covers what you can verify without a GPU and how to
test the real inference path on GPU hardware.

## Tier 0 — no GPU needed (runs anywhere)

Already covered by the committed test suite:

```bash
npm test
```

- `resolveAsr` branch coverage: GPU present/absent, `TRANSCRIPTION_PROVIDER`
  overrides, `NEMO_ENABLED=false`, custom `NEMO_MODEL`.
- The Node ↔ sidecar contract (`transcribeWithNemo`): temp-file handoff, JSON
  parsing, and non-zero-exit error surfacing, via stub interpreters.

Graceful degradation is also observable live on a non-GPU box:

```bash
TRANSCRIPTION_PROVIDER=nemo npm start
curl -s localhost:3000/api/health
# → "asr": { "provider": "mock", "gpu": false,
#            "reason": "NeMo requested but no NVIDIA GPU detected" }
```

## Tier 1 — quick sidecar check on a free Colab T4

The sidecar is self-contained, so the GPU inference contract can be proven in
minutes on [Google Colab](https://colab.research.google.com) (Runtime → T4 GPU):

```python
!nvidia-smi -L                     # confirm the GPU
!pip install -q "nemo_toolkit[asr]"
!wget -q https://raw.githubusercontent.com/AntonioZZ841/Whisper-Flow-clone/claude/whisper-flow-clone-411bt3/scripts/nemo_transcribe.py
# upload any audio clip via the Files panel, then:
!python3 nemo_transcribe.py your_clip.m4a
# → {"text": "..."}   ← the exact contract src/asr.js consumes
```

## Tier 2 — full app on a GPU machine

Your own PC with an NVIDIA card, or a rented VM (RunPod / Lambda /
AWS `g4dn.xlarge`):

```bash
nvidia-smi                          # 1. GPU visible?

git clone https://github.com/AntonioZZ841/Whisper-Flow-clone.git
cd Whisper-Flow-clone
npm install

python3 -m venv .venv               # 2. Python side (venv recommended)
.venv/bin/pip install "nemo_toolkit[asr]"     # pulls CUDA-enabled torch
sudo apt install -y ffmpeg

# 3. Sanity-check the sidecar alone before the full app:
.venv/bin/python3 scripts/nemo_transcribe.py sample.wav

# 4. Full app, pointing the sidecar at the venv interpreter:
NEMO_PYTHON=.venv/bin/python3 TRANSCRIPTION_PROVIDER=nemo npm start
```

If the GPU box is remote, tunnel so your browser microphone works (mic capture
requires a secure context, and `localhost` through a tunnel qualifies):

```bash
ssh -L 3000:localhost:3000 user@gpu-box    # then open http://localhost:3000
```

## Pass criteria

1. Startup log: `Stage 1 ASR:  nemo (nvidia/parakeet-tdt-0.6b-v2) · GPU detected`
2. `curl localhost:3000/api/health` → `"asr": { "provider": "nemo", ..., "gpu": true }`
3. Speak or upload a clip → the **raw pane shows your actual words**.

Expectations: the first request downloads the ~600 MB model; the sidecar also
reloads the model on every call by design (simple, not fast). For a warm-model
deployment, run a resident NeMo service exposing an OpenAI-compatible
`/audio/transcriptions` endpoint and use the `openai-compatible` provider
pointed at it — see the note in `scripts/nemo_transcribe.py`.
