"""Group raw whisper segments into longer, time-bounded chunks."""

from __future__ import annotations

from typing import List

from .models import TranscriptChunk


def merge_segments(
    segments: List[TranscriptChunk],
    target_seconds: float = 60.0,
    max_seconds: float = 90.0,
) -> List[TranscriptChunk]:
    """Greedy merge of adjacent segments until each chunk approaches `target_seconds`.

    A chunk is closed when adding the next segment would exceed `max_seconds`,
    or when the natural end of input is reached.
    """
    if not segments:
        return []

    merged: List[TranscriptChunk] = []
    buf_start = segments[0].start
    buf_end = segments[0].end
    buf_text: List[str] = [segments[0].text]

    for seg in segments[1:]:
        prospective = seg.end - buf_start
        if prospective <= max_seconds and (buf_end - buf_start) < target_seconds:
            buf_end = seg.end
            buf_text.append(seg.text)
        else:
            merged.append(TranscriptChunk(buf_start, buf_end, " ".join(buf_text).strip()))
            buf_start, buf_end, buf_text = seg.start, seg.end, [seg.text]

    merged.append(TranscriptChunk(buf_start, buf_end, " ".join(buf_text).strip()))
    return merged
