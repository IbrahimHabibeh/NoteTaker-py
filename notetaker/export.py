"""Render notes to Markdown and dump the raw transcript to JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from .models import Notes, TranscriptChunk, _fmt


def export_markdown(notes: Notes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# {notes.title}")
    lines.append("")
    lines.append(f"_Source: {notes.source}_  ")
    lines.append(f"_Duration: {_fmt(notes.duration)}_")
    lines.append("")

    for section in notes.sections:
        lines.append(f"## {section.title}")
        lines.append(f"_{_fmt(section.start)}–{_fmt(section.end)}_")
        lines.append("")
        for chunk in section.chunks:
            lines.append(f"- **[{chunk.timestamp()}]** {chunk.summary}")
            for bullet in chunk.bullets:
                if bullet.strip() and bullet.strip() != chunk.summary.strip():
                    lines.append(f"    - {bullet}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def export_transcript_json(chunks: List[TranscriptChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(c) for c in chunks]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_notes_json(notes: Notes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(notes.model_dump_json(indent=2), encoding="utf-8")
