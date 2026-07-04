#!/usr/bin/env python3
"""Transcribe one audio file with an NVIDIA NeMo ASR model.

Reads the audio path from argv[1], prints a single JSON line: {"text": "..."}.
Invoked per utterance by src/asr.js when the `nemo` provider is active.

Requirements (on the deployment server):
  - an NVIDIA GPU with CUDA
  - pip install "nemo_toolkit[asr]"
  - ffmpeg on PATH (browser clips are webm/opus; NeMo wants 16 kHz mono WAV)

Config via env:
  NEMO_MODEL   pretrained model name or .nemo path
               (default: nvidia/parakeet-tdt-0.6b-v2)

Note: this loads the model on every call, which is simple but not fast. For a
busy deployment, run a resident NeMo service exposing an OpenAI-compatible
/audio/transcriptions endpoint and point TRANSCRIPTION_API_URL at it instead
(the `openai-compatible` provider) so the model stays warm.
"""

import json
import os
import subprocess
import sys
import tempfile

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"


def fail(message):
    print(json.dumps({"error": message}), file=sys.stderr)
    sys.exit(1)


def to_wav_16k_mono(src):
    """Transcode any input to 16 kHz mono WAV via ffmpeg."""
    fd, dst = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", dst],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        os.remove(dst)
        fail("ffmpeg not found on PATH (needed to decode the audio for NeMo)")
    except subprocess.CalledProcessError as exc:
        os.remove(dst)
        fail(f"ffmpeg failed: {exc.stderr.decode('utf-8', 'ignore')[-300:]}")
    return dst


def extract_text(result):
    """NeMo's transcribe() return shape varies across versions/models:
    a list of strings, or a list of Hypothesis objects with a .text attr."""
    if not result:
        return ""
    first = result[0]
    return getattr(first, "text", first)


def main():
    if len(sys.argv) < 2:
        fail("usage: nemo_transcribe.py <audio-file>")

    model_name = os.environ.get("NEMO_MODEL", DEFAULT_MODEL)
    wav = to_wav_16k_mono(sys.argv[1])

    try:
        import nemo.collections.asr as nemo_asr  # heavy import; keep it lazy
    except ImportError:
        os.remove(wav)
        fail('nemo_toolkit not installed — run: pip install "nemo_toolkit[asr]"')

    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        result = model.transcribe([wav])
        print(json.dumps({"text": str(extract_text(result))}))
    except Exception as exc:  # noqa: BLE001 — surface any model/runtime error
        fail(f"NeMo transcription failed: {exc}")
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass


if __name__ == "__main__":
    main()
