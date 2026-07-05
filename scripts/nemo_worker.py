#!/usr/bin/env python3
"""Resident NeMo worker: load the models once, serve many transcriptions.

Spawned by src/asr.js (the default; NEMO_KEEP_WARM=false reverts to the
one-shot nemo_transcribe.py). Speaks JSON lines over stdio:

  ← {"ready": true}                                   on startup
  → {"id": 1, "warm": true}                           preload the ASR model
  ← {"id": 1, "ok": true}
  → {"id": 2, "path": "/tmp/clip.webm", "diarize": false}
  ← {"id": 2, "text": "...", "turns": [...]?, "warning": "..."?}
  ← {"id": 2, "error": "..."}                         on per-request failure

Models are cached in-process (ASR on first use, the Sortformer diarizer on
the first diarize request) and stay on the GPU between requests — this is
what removes the ~15 s per-call model reload of the one-shot sidecar.
Requests are handled sequentially; a per-request failure is answered, never
fatal. Exits when stdin closes (the Node process died).

Config via env: NEMO_MODEL, NEMO_DIAR_MODEL — see nemo_transcribe.py.
"""

import json
import os
import sys

from nemo_transcribe import (
    DEFAULT_MODEL,
    diarize_turns,
    extract_speaker_segments,
    extract_text,
    extract_word_spans,
    to_wav_16k_mono,
)

# Our stdout contract is one JSON line per message, but NeMo's logger writes
# to stdout. Keep the real stream for ourselves, give the logger stderr.
OUT = sys.stdout
sys.stdout = sys.stderr

_asr_model = None
_diar_model = None


def send(obj):
    print(json.dumps(obj), file=OUT, flush=True)


def _disable_cuda_graph_decoding(model):
    """Parakeet's CUDA-graph-captured decoding loop corrupts CUDA state when
    another model (the Sortformer diarizer) runs kernels between captures:
    the next transcribe dies with 'illegal memory access' in
    currentStreamCaptureStatusMayInitCtx and SIGABRTs the process. Eager
    decoding costs ~0.1 s on dictation-length clips and is stable, so turn
    the graph decoder off for the resident worker."""
    try:
        from omegaconf import open_dict

        decoding = model.cfg.decoding
        with open_dict(decoding):
            if "greedy" in decoding:
                decoding.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(decoding)
    except Exception as exc:  # noqa: BLE001 — degraded stability, not fatal
        print(f"note: could not disable CUDA-graph decoding: {exc}", file=sys.stderr)


def get_asr():
    global _asr_model
    if _asr_model is None:
        import nemo.collections.asr as nemo_asr  # heavy import; keep it lazy

        _asr_model = nemo_asr.models.ASRModel.from_pretrained(
            os.environ.get("NEMO_MODEL", DEFAULT_MODEL)
        )
        _disable_cuda_graph_decoding(_asr_model)
    return _asr_model


def get_diar():
    global _diar_model
    if _diar_model is None:
        from nemo_transcribe import load_diar_model

        _diar_model = load_diar_model()
    return _diar_model


def handle(req):
    rid = req.get("id")
    try:
        if req.get("warm"):
            get_asr()
            send({"id": rid, "ok": True})
            return

        # Speaker segments only — no ASR. Used by the hybrid meeting mode,
        # where a multilingual Whisper server supplies the words and this
        # (language-agnostic) diarizer supplies who spoke when.
        if req.get("segments_only"):
            wav = to_wav_16k_mono(req["path"])
            try:
                segments = extract_speaker_segments(get_diar().diarize(audio=[wav], batch_size=1))
                send({"id": rid, "segments": segments})
            finally:
                try:
                    os.remove(wav)
                except OSError:
                    pass
            return

        diarize = bool(req.get("diarize"))
        wav = to_wav_16k_mono(req["path"])
        try:
            result = get_asr().transcribe([wav], timestamps=diarize)
            out = {"id": rid, "text": str(extract_text(result))}
            if diarize:
                try:
                    turns, warning = diarize_turns(wav, extract_word_spans(result), get_diar())
                    out["turns"] = turns
                    if warning:
                        out["warning"] = warning
                except Exception as exc:  # noqa: BLE001 — degrade, keep the text
                    out["turns"] = None
                    out["warning"] = f"speaker labeling failed: {exc}"
            send(out)
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass
    except Exception as exc:  # noqa: BLE001 — answer the request, stay alive
        send({"id": rid, "error": str(exc)[:300]})


def main():
    send({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if isinstance(req, dict):
            handle(req)


if __name__ == "__main__":
    main()
