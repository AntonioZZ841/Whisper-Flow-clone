#!/usr/bin/env python3
"""Local Whisper transcription server (OpenAI-compatible).

Exposes POST /v1/audio/transcriptions (multipart form: `file`, optional
`language`) returning {"text": ...} — the exact surface the app's
`openai-compatible` ASR provider expects, so pointing TRANSCRIPTION_API_URL
here gives fully local, multilingual dictation (Mandarin, English, and
code-switched speech included) with the model resident on the GPU.

  python scripts/whisper_server.py          # or the venv's python

Config via env:
  WHISPER_MODEL    faster-whisper model name (default: large-v3)
  WHISPER_PORT     listen port (default: 8756)
  WHISPER_DEVICE   cuda | cpu (default: cuda)

Requirements (a dedicated venv is fine):
  pip install faster-whisper flask "nvidia-cudnn-cu12>=9" nvidia-cublas-cu12

The first run downloads the model (~3 GB for large-v3) to the local cache.
faster-whisper runs on CTranslate2, which needs the pip-installed cuDNN and
cuBLAS libraries on LD_LIBRARY_PATH — this script fixes that up itself by
re-exec'ing once with the corrected environment, so no wrapper is needed.
"""

import json
import os
import sys
import tempfile


def ensure_cuda_libs():
    """CTranslate2 dlopens libcudnn/libcublas, which pip installs outside the
    default loader path. The dynamic loader only reads LD_LIBRARY_PATH at
    process start, so if the libs aren't reachable yet, re-exec once with the
    corrected environment."""
    if os.environ.get("_WHISPER_SERVER_REEXEC") == "1":
        return
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
    except ImportError:
        return  # CPU-only install — nothing to fix
    # These are namespace packages: __file__ is None, the directory lives in
    # __path__.
    lib_dirs = [
        next(iter(nvidia.cublas.lib.__path__)),
        next(iter(nvidia.cudnn.lib.__path__)),
    ]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    missing = [d for d in lib_dirs if d not in current.split(":")]
    if missing:
        os.environ["LD_LIBRARY_PATH"] = ":".join(missing + ([current] if current else []))
        os.environ["_WHISPER_SERVER_REEXEC"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


ensure_cuda_libs()

from faster_whisper import WhisperModel  # noqa: E402
from flask import Flask, jsonify, request  # noqa: E402

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
PORT = int(os.environ.get("WHISPER_PORT", "8756"))
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")

print(f"loading {MODEL_NAME} on {DEVICE} (first run downloads the model)...", flush=True)
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")
print("model loaded", flush=True)

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"ok": True, "model": MODEL_NAME, "device": DEVICE})


@app.post("/v1/audio/transcriptions")
def transcribe():
    upload = request.files.get("file")
    if upload is None:
        return jsonify({"error": "no `file` field in the multipart form"}), 400
    language = request.form.get("language") or None  # None = auto-detect
    # Extension beyond the OpenAI surface: per-word timestamps, used by the
    # app's hybrid meeting mode to merge words with speaker segments.
    want_words = request.form.get("word_timestamps") == "true"

    suffix = os.path.splitext(upload.filename or "")[1] or ".webm"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            upload.save(f)
        # vad_filter trims leading/trailing silence, which also keeps Whisper
        # from hallucinating text for silent or non-speech clips.
        segments, info = model.transcribe(
            path, language=language, vad_filter=True, word_timestamps=want_words
        )
        out_words = []
        texts = []
        for segment in segments:  # generator — consume once
            texts.append(segment.text)
            if want_words:
                for w in segment.words or []:
                    out_words.append(
                        {"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2)}
                    )
        result = {"text": "".join(texts).strip(), "language": info.language}
        if want_words:
            result["words"] = out_words
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001 — answer the request, stay alive
        return jsonify({"error": str(exc)[:300]}), 500
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    print(f"whisper server → http://127.0.0.1:{PORT}/v1/audio/transcriptions", flush=True)
    app.run(host="127.0.0.1", port=PORT, threaded=False)
