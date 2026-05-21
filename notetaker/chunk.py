"""Group raw whisper segments into longer, time-bounded chunks."""

from __future__ import annotations

from typing import List

from .models import TranscriptChunk


def merge_segments(
    segments: List[TranscriptChunk],
    target_seconds: float = 60.0,
    max_seconds: float = 90.0,
    pause_seconds: float = 0.85,
    target_grace_seconds: float = 12.0,
) -> List[TranscriptChunk]:
    """Merge adjacent segments into chunks with soft-boundary aware closure.

    Strategy:
    - Prefer closing after sentence endings or noticeable pauses once a chunk
      reaches `target_seconds`.
    - If no good boundary appears, allow a small overrun (`target_grace_seconds`).
    - Never exceed `max_seconds`; when forced, close at the latest soft boundary
      within the chunk when possible.
    """
    if not segments:
        return []

    merged: List[TranscriptChunk] = []
    start_idx = 0
    i = 0
    last_soft_idx: int | None = None

    while i < len(segments) - 1:
        current = segments[i]
        nxt = segments[i + 1]

        soft_here = _is_soft_boundary(current, nxt, pause_seconds)
        if soft_here:
            last_soft_idx = i

        cur_duration = _duration(segments, start_idx, i)
        next_duration = _duration(segments, start_idx, i + 1)

        # Hard cap: if next addition would exceed max, close now (prefer last soft).
        if next_duration > max_seconds:
            end_idx = _best_forced_end(start_idx, i, last_soft_idx, segments)
            merged.append(_make_chunk(segments, start_idx, end_idx))
            start_idx = end_idx + 1
            i = start_idx
            last_soft_idx = None
            continue

        # Natural closure once target is reached and a soft boundary appears.
        if soft_here and cur_duration >= target_seconds:
            merged.append(_make_chunk(segments, start_idx, i))
            start_idx = i + 1
            i = start_idx
            last_soft_idx = None
            continue

        # If we drift too far beyond target, close at best known boundary.
        if cur_duration >= target_seconds + target_grace_seconds:
            end_idx = _best_forced_end(start_idx, i, last_soft_idx, segments)
            merged.append(_make_chunk(segments, start_idx, end_idx))
            start_idx = end_idx + 1
            i = start_idx
            last_soft_idx = None
            continue

        i += 1

    if start_idx < len(segments):
        merged.append(_make_chunk(segments, start_idx, len(segments) - 1))

    return merged


def _make_chunk(segments: List[TranscriptChunk], start_idx: int, end_idx: int) -> TranscriptChunk:
    text = " ".join(seg.text for seg in segments[start_idx : end_idx + 1]).strip()
    return TranscriptChunk(
        start=segments[start_idx].start,
        end=segments[end_idx].end,
        text=text,
    )


def _duration(segments: List[TranscriptChunk], start_idx: int, end_idx: int) -> float:
    return segments[end_idx].end - segments[start_idx].start


def _best_forced_end(
    start_idx: int,
    current_idx: int,
    last_soft_idx: int | None,
    segments: List[TranscriptChunk],
) -> int:
    # Prefer a previously seen soft boundary if it is meaningfully long.
    if last_soft_idx is not None and last_soft_idx >= start_idx:
        if _duration(segments, start_idx, last_soft_idx) >= 25.0:
            return last_soft_idx
    return current_idx


def _is_soft_boundary(prev: TranscriptChunk, nxt: TranscriptChunk, pause_seconds: float) -> bool:
    prev_text = prev.text.strip()
    gap = max(0.0, nxt.start - prev.end)

    # A real pause often indicates thought completion.
    if gap >= pause_seconds:
        return True

    # Sentence-ending punctuation is a strong boundary signal.
    if prev_text.endswith((".", "!", "?")):
        return True

    # Semicolons/colons are weaker; combine with a brief pause.
    if prev_text.endswith((";", ":")) and gap >= pause_seconds * 0.45:
        return True

    return False
