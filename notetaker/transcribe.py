"""Whisper-based transcription producing timestamped chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .models import TranscriptChunk


def transcribe(
    audio: Path,
    model_size: str = "base.en",
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
) -> List[TranscriptChunk]:
    """Transcribe `audio` with faster-whisper and return timestamped segments."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(audio),
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    return _to_chunks(segments)


def _to_chunks(segments: Iterable) -> List[TranscriptChunk]:
    chunks: List[TranscriptChunk] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        chunks.append(TranscriptChunk(start=float(seg.start), end=float(seg.end), text=text))
    return chunks
