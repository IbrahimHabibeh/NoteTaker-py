"""Audio acquisition via yt-dlp (URLs) or direct local files."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen

from .models import TranscriptChunk


@dataclass
class TranscriptFetchResult:
    """Result of attempting to fetch a native YouTube transcript."""

    title: str
    segments: List[TranscriptChunk]
    source_kind: str  # manual-subtitles | automatic-captions


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "www."))


def try_fetch_youtube_transcript(
    source: str,
    language: str | None = None,
) -> Optional[TranscriptFetchResult]:
    """Try to fetch timestamped transcript directly from YouTube via yt-dlp.

    Returns None when no usable subtitle/caption track is available.
    """
    if not is_url(source):
        return None

    from yt_dlp import YoutubeDL

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=False)

    title = info.get("title") or info.get("id") or "audio"
    subtitles = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}

    pick = _pick_caption_track(subtitles, automatic, language)
    if not pick:
        return None

    raw = _download_text(pick["url"])
    segments = _parse_caption_payload(raw, pick["ext"])
    if not segments:
        return None

    kind = "manual-subtitles" if pick["origin"] == "subtitles" else "automatic-captions"
    return TranscriptFetchResult(title=title, segments=segments, source_kind=kind)


def download_audio(source: str, out_dir: Path) -> Tuple[Path, str]:
    """Resolve `source` to a local audio file and return (path, title).

    If `source` is a URL, download the best audio track via yt-dlp.
    If `source` is a local file, return it as-is.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not is_url(source):
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return path, path.stem

    # Lazy import: yt-dlp is only needed for URL inputs.
    from yt_dlp import YoutubeDL

    outtmpl = str(out_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=True)
        title = info.get("title") or info.get("id") or "audio"
        audio_path = out_dir / f"{info['id']}.mp3"

    if not audio_path.exists():
        raise RuntimeError(f"yt-dlp did not produce expected file: {audio_path}")
    return audio_path, title


def _pick_caption_track(
    subtitles: Dict,
    automatic: Dict,
    language: str | None,
) -> Optional[Dict[str, str]]:
    for origin, tracks in (("subtitles", subtitles), ("automatic_captions", automatic)):
        if not tracks:
            continue
        lang_keys = _rank_language_keys(list(tracks.keys()), language)
        for lang in lang_keys:
            items = tracks.get(lang) or []
            best = _best_item(items)
            if best:
                return {
                    "origin": origin,
                    "lang": lang,
                    "ext": best.get("ext", "vtt"),
                    "url": best["url"],
                }
    return None


def _rank_language_keys(keys: List[str], preferred: str | None) -> List[str]:
    if not keys:
        return []

    lowered = {k.lower(): k for k in keys}
    ordered: List[str] = []

    def add(k: str) -> None:
        if k in keys and k not in ordered:
            ordered.append(k)

    if preferred:
        p = preferred.lower()
        if p in lowered:
            add(lowered[p])
        for k in keys:
            lk = k.lower()
            if lk.startswith(p + "-") or lk.startswith(p + "_"):
                add(k)

    for en in ("en", "en-us", "en-gb"):
        if en in lowered:
            add(lowered[en])

    for k in sorted(keys):
        add(k)
    return ordered


def _best_item(items: List[Dict]) -> Optional[Dict]:
    if not items:
        return None
    # Prefer json3 because it carries clean timestamps; then vtt/srv3/ttml.
    ext_rank = {"json3": 0, "vtt": 1, "srv3": 2, "ttml": 3}
    ranked = sorted(
        [it for it in items if it.get("url")],
        key=lambda it: ext_rank.get((it.get("ext") or "").lower(), 99),
    )
    return ranked[0] if ranked else None


def _download_text(url: str) -> str:
    with urlopen(url) as res:  # nosec B310 - URL comes from yt-dlp metadata
        data = res.read()
    return data.decode("utf-8", errors="replace")


def _parse_caption_payload(payload: str, ext: str) -> List[TranscriptChunk]:
    ext = (ext or "").lower()
    if ext == "json3":
        return _parse_json3(payload)
    if ext in {"vtt", "srv3", "ttml"}:
        # srv3/ttml fallback through text parser still catches many tracks.
        return _parse_vtt_like(payload)

    # Unknown ext: try both parsers defensively.
    chunks = _parse_json3(payload)
    if chunks:
        return chunks
    return _parse_vtt_like(payload)


def _parse_json3(payload: str) -> List[TranscriptChunk]:
    try:
        obj = json.loads(payload)
    except Exception:
        return []

    events = obj.get("events") or []
    out: List[TranscriptChunk] = []
    for ev in events:
        segs = ev.get("segs") or []
        if not segs:
            continue

        text = "".join((seg.get("utf8") or "") for seg in segs)
        text = _clean_caption_text(text)
        if not text:
            continue

        start_ms = ev.get("tStartMs")
        dur_ms = ev.get("dDurationMs")
        if start_ms is None:
            continue

        start = float(start_ms) / 1000.0
        end = start + max(float(dur_ms or 0), 500.0) / 1000.0
        out.append(TranscriptChunk(start=start, end=end, text=text))

    return _squash_duplicates(out)


def _parse_vtt_like(payload: str) -> List[TranscriptChunk]:
    lines = payload.splitlines()
    out: List[TranscriptChunk] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" not in line:
            i += 1
            continue

        try:
            left, right = [p.strip() for p in line.split("-->", 1)]
            start = _parse_vtt_timestamp(left)
            end = _parse_vtt_timestamp(right.split()[0])
        except Exception:
            i += 1
            continue

        i += 1
        cue_text: List[str] = []
        while i < len(lines) and lines[i].strip():
            cue_text.append(lines[i].strip())
            i += 1

        text = _clean_caption_text(" ".join(cue_text))
        if text:
            out.append(TranscriptChunk(start=start, end=end, text=text))

        i += 1

    return _squash_duplicates(out)


def _parse_vtt_timestamp(ts: str) -> float:
    # Supports HH:MM:SS.mmm and MM:SS.mmm.
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def _clean_caption_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _squash_duplicates(chunks: List[TranscriptChunk]) -> List[TranscriptChunk]:
    if not chunks:
        return []
    out: List[TranscriptChunk] = [chunks[0]]
    for cur in chunks[1:]:
        prev = out[-1]
        if cur.text == prev.text and cur.start <= prev.end + 0.25:
            prev.end = max(prev.end, cur.end)
            continue
        out.append(cur)
    return out
