#!/usr/bin/env python3
"""Transcribe one audio file with an NVIDIA NeMo ASR model.

Usage: nemo_transcribe.py <audio-file> [--diarize]

Prints a single JSON line. Plain mode: {"text": "..."}. With --diarize, the
audio is additionally segmented by speaker (NeMo Sortformer, up to 4
speakers) and word timestamps from the ASR pass are merged with the speaker
segments into turns:

  {"text": "...", "turns": [{"speaker": "Speaker 1", "start": 0.0,
                             "end": 3.2, "text": "..."}, ...]}

If diarization fails (model unavailable, no speech, ...) the transcript is
still returned, with "turns": null and a "warning" — labeling degrades,
dictation is never lost. Invoked per utterance by src/asr.js when the `nemo`
provider is active.

Requirements (on the deployment server):
  - an NVIDIA GPU with CUDA
  - pip install "nemo_toolkit[asr]"
  - ffmpeg on PATH (browser clips are webm/opus; NeMo wants 16 kHz mono WAV)

Config via env:
  NEMO_MODEL        pretrained ASR model name or .nemo path
                    (default: nvidia/parakeet-tdt-0.6b-v2)
  NEMO_DIAR_MODEL   diarization model (default: nvidia/diar_sortformer_4spk-v1)

Note: this loads the model(s) on every call, which is simple but not fast.
For a busy deployment, run a resident NeMo service exposing an
OpenAI-compatible /audio/transcriptions endpoint and point
TRANSCRIPTION_API_URL at it instead (the `openai-compatible` provider) so the
models stay warm.
"""

import json
import os
import subprocess
import sys
import tempfile

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
DEFAULT_DIAR_MODEL = "nvidia/diar_sortformer_4spk-v1"

# Encoder frame duration used to convert timestamp offsets to seconds when a
# NeMo version reports offsets instead of seconds (FastConformer: 80 ms).
FRAME_SECONDS = 0.08


def fail(message):
    print(json.dumps({"error": message}), file=sys.stderr)
    sys.exit(1)


def to_wav_16k_mono(src):
    """Transcode any input to 16 kHz mono WAV via ffmpeg. Raises RuntimeError
    on failure so long-lived callers (nemo_worker.py) survive a bad file."""
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
        raise RuntimeError("ffmpeg not found on PATH (needed to decode the audio for NeMo)")
    except subprocess.CalledProcessError as exc:
        os.remove(dst)
        raise RuntimeError(f"ffmpeg failed: {exc.stderr.decode('utf-8', 'ignore')[-300:]}")
    return dst


def extract_text(result):
    """NeMo's transcribe() return shape varies across versions/models:
    a list of strings, or a list of Hypothesis objects with a .text attr."""
    if not result:
        return ""
    first = result[0]
    return getattr(first, "text", first)


def extract_word_spans(result):
    """Word-level (start, end, word) tuples from a timestamps=True transcribe
    result. Tolerates seconds ('start'/'end') or encoder-frame offsets."""
    if not result:
        return []
    ts = getattr(result[0], "timestamp", None) or {}
    spans = []
    for w in ts.get("word") or []:
        if not isinstance(w, dict):
            continue
        word = str(w.get("word", "")).strip()
        start, end = w.get("start"), w.get("end")
        if start is None and w.get("start_offset") is not None:
            start = w["start_offset"] * FRAME_SECONDS
            end = (w.get("end_offset") or w["start_offset"]) * FRAME_SECONDS
        if not word or start is None:
            continue
        spans.append((float(start), float(end if end is not None else start), word))
    return spans


def extract_speaker_segments(diar_result):
    """(start, end, label) tuples from Sortformer's diarize() output — a list
    per file of 'start end speaker_k' strings (tolerating list/tuple forms)."""
    if not diar_result:
        return []
    per_file = diar_result[0]
    segments = []
    for seg in per_file or []:
        if isinstance(seg, str):
            parts = seg.split()
            if len(parts) < 3:
                continue
            start, end, label = parts[0], parts[1], parts[2]
        elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
            start, end, label = seg[0], seg[1], seg[2]
        else:
            continue
        try:
            segments.append((float(start), float(end), str(label)))
        except (TypeError, ValueError):
            continue
    return sorted(segments, key=lambda s: s[0])


def merge_into_turns(word_spans, segments):
    """Assign each word to the speaker segment covering its midpoint (nearest
    segment when none covers it), then merge consecutive same-speaker words
    into turns with human-friendly labels in order of first appearance."""
    if not word_spans or not segments:
        return None

    def speaker_at(t):
        for start, end, label in segments:
            if start <= t <= end:
                return label
        return min(segments, key=lambda s: min(abs(s[0] - t), abs(s[1] - t)))[2]

    turns = []
    for start, end, word in word_spans:
        label = speaker_at((start + end) / 2)
        if turns and turns[-1]["label"] == label:
            turns[-1]["end"] = end
            turns[-1]["words"].append(word)
        else:
            turns.append({"label": label, "start": start, "end": end, "words": [word]})

    names = {}
    for t in turns:
        names.setdefault(t["label"], f"Speaker {len(names) + 1}")
    return [
        {
            "speaker": names[t["label"]],
            "start": round(t["start"], 2),
            "end": round(t["end"], 2),
            "text": " ".join(t["words"]),
        }
        for t in turns
    ]


def load_diar_model():
    from nemo.collections.asr.models import SortformerEncLabelModel

    return SortformerEncLabelModel.from_pretrained(
        os.environ.get("NEMO_DIAR_MODEL", DEFAULT_DIAR_MODEL)
    )


def diarize_turns(wav, word_spans, diar_model=None):
    """Speaker turns for `wav`, or (None, warning) when labeling is not
    possible — diarization failure must never lose the transcript. Pass a
    preloaded model (nemo_worker.py) to skip the per-call load."""
    diar_model = diar_model or load_diar_model()
    segments = extract_speaker_segments(diar_model.diarize(audio=[wav], batch_size=1))
    turns = merge_into_turns(word_spans, segments)
    if not turns:
        return None, "no speaker segments detected — returning an unlabeled transcript"
    return turns, None


def main():
    if len(sys.argv) < 2:
        fail("usage: nemo_transcribe.py <audio-file> [--diarize]")
    diarize = "--diarize" in sys.argv[2:]

    model_name = os.environ.get("NEMO_MODEL", DEFAULT_MODEL)
    try:
        wav = to_wav_16k_mono(sys.argv[1])
    except RuntimeError as exc:
        fail(str(exc))

    # NeMo's logger writes to stdout, but our contract with src/asr.js is that
    # stdout carries exactly one JSON line. Point sys.stdout at stderr before
    # the import (the logger binds its stream then), restore it for the result.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        import nemo.collections.asr as nemo_asr  # heavy import; keep it lazy
    except ImportError:
        os.remove(wav)
        fail('nemo_toolkit not installed — run: pip install "nemo_toolkit[asr]"')

    turns = None
    warning = None
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        result = model.transcribe([wav], timestamps=diarize)
        text = str(extract_text(result))
        if diarize:
            try:
                turns, warning = diarize_turns(wav, extract_word_spans(result))
            except Exception as exc:  # noqa: BLE001 — degrade, don't lose the text
                warning = f"speaker labeling failed: {exc}"
    except Exception as exc:  # noqa: BLE001 — surface any model/runtime error
        fail(f"NeMo transcription failed: {exc}")
    finally:
        sys.stdout = real_stdout
        try:
            os.remove(wav)
        except OSError:
            pass

    out = {"text": text}
    if diarize:
        out["turns"] = turns
        if warning:
            out["warning"] = warning
    print(json.dumps(out))


if __name__ == "__main__":
    main()
