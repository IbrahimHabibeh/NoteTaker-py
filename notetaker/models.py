"""Typed data models shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, Field


@dataclass
class TranscriptChunk:
    """A contiguous, timestamped slice of transcript text."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def timestamp(self) -> str:
        return f"{_fmt(self.start)}–{_fmt(self.end)}"


class ChunkSummary(BaseModel):
    """A transcript chunk plus its extracted highlights."""

    start: float
    end: float
    text: str
    summary: str
    bullets: List[str] = Field(default_factory=list)

    def timestamp(self) -> str:
        return f"{_fmt(self.start)}–{_fmt(self.end)}"


class TopicSection(BaseModel):
    """A topic-level grouping of consecutive chunks."""

    title: str
    chunks: List[ChunkSummary]

    @property
    def start(self) -> float:
        return self.chunks[0].start if self.chunks else 0.0

    @property
    def end(self) -> float:
        return self.chunks[-1].end if self.chunks else 0.0


class Notes(BaseModel):
    """Top-level hierarchical notes document."""

    title: str
    source: str
    duration: float
    sections: List[TopicSection]


def _fmt(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"
